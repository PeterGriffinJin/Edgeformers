import logging
import os
import pickle
import random
from time import time
from collections import defaultdict
from sklearn.metrics import f1_score

from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from src.utils import setuplogging
from src.data_bert import load_dataset_bert
from transformers import BertConfig, BertTokenizerFast, AdamW, get_linear_schedule_with_warmup

from transformers import BertModel

def cleanup():
    dist.destroy_process_group()

def load_bert(args):
    config = BertConfig.from_pretrained(args.model_name_or_path, output_hidden_states=True)
    config.heter_embed_size = args.heter_embed_size
    config.node_num = args.user_num + args.item_num
    config.class_num = args.class_num
    args.hidden_size = config.hidden_size
    args.node_num = args.user_num + args.item_num
    if args.model_type == 'EdgeformerE':
        from src.model.EdgeformerE import EdgeFormersForEdgeClassification
        model = EdgeFormersForEdgeClassification.from_pretrained(args.model_name_or_path, config=config) if args.pretrain_LM else EdgeFormersForEdgeClassification(config)
        model.node_num, model.edge_type, model.heter_embed_size = args.user_num + args.item_num, args.class_num, args.heter_embed_size
        model.init_node_embed(args.pretrain_embed, args.pretrain_mode, args.pretrain_dir)
    else:
        raise ValueError('Input Model Name is Incorrect!')

    return model

def train(args):

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # config.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    logging.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))

    # Load data
    # define tokenizer 
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # load dataset
    if args.data_mode in ['bert']:
        args.user_num, args.item_num, args.class_num = pickle.load(open(os.path.join(args.data_path, 'node_num.pkl'),'rb'))
        train_set = load_dataset_bert(args, tokenizer, evaluate=False, test=False)
        val_set = load_dataset_bert(args, tokenizer, evaluate=True, test=False)
        test_set = load_dataset_bert(args, tokenizer, evaluate=True, test=True)
    else:
        raise ValueError('Data Mode is Incorrect here!')

    # define dataloader
    train_sampler = RandomSampler(train_set) if args.local_rank == -1 else DistributedSampler(train_set)
    val_sampler = SequentialSampler(val_set) if args.local_rank == -1 else DistributedSampler(val_set)
    test_sampler = SequentialSampler(test_set) if args.local_rank == -1 else DistributedSampler(test_set)

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, sampler=val_sampler)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, sampler=test_sampler)
    print(f'[Process:{args.local_rank}] Dataset Loading Over!')

    # define model
    model = load_bert(args)
    if args.local_rank in [-1, 0]:
        logging.info('loading model: {}'.format(args.model_type))
    model.to(args.device)

    if args.load:
        model.load_state_dict(torch.load(args.load_ckpt_name, map_location="cpu"))
        logging.info('load ckpt:{}'.format(args.load_ckpt_name))

    # define DDP here
    if args.local_rank != -1:
        ddp_model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    else:
        ddp_model = model

    # define optimizer
    ###################### You should think more here about the weight_decay and adam_epsilon ##################
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    # optimizer = optim.Adam([{'params': ddp_model.parameters(), 'lr': args.lr}])
    # t_total = len(train_loader) * args.epochs // (args.train_batch_size) ################### be careful, no DDP here!!! ###################
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    #train
    loss = 0.0
    global_step = 0
    best_acc, best_count = 0.0, 0

    for ep in range(args.epochs):
        ## start training
        start_time = time()
        ddp_model.train() ######################## You should motify it to ddp_model.train when using DDP
        train_loader_iterator = tqdm(train_loader, desc=f"Epoch:{ep}|Iteration", disable=args.local_rank not in [-1,0])
        for step, batch in enumerate(train_loader_iterator):
            # put data into GPU
            if args.enable_gpu:
                batch = [b.cuda() for b in batch]

            # calculate loss
            batch_loss = ddp_model(*batch)
            loss += batch_loss.item()
            optimizer.zero_grad()
            batch_loss.backward()
            # torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), args.max_grad_norm)
            optimizer.step()
            # scheduler.step()
            global_step += 1

            # logging
            if args.local_rank in [-1, 0]:
                if global_step % args.log_steps == 0:
                    logging.info(
                        'cost_time:{} step:{}, lr:{}, train_loss: {:.5f}'.format(
                            time() - start_time, global_step, optimizer.param_groups[0]['lr'],
                            loss / args.log_steps))
                    loss = 0.0
                if args.local_rank == 0:
                    torch.distributed.barrier()
            else:
                torch.distributed.barrier()

        ## start validating
        if args.local_rank in [-1, 0]:
            ckpt_path = os.path.join(args.data_path, 'ckpt', '{}-{}-{}-{}-epoch-{}.pt'.format(args.model_type, args.pretrain_LM, args.lr, args.heter_embed_size, ep + 1))
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"Model saved to {ckpt_path}")

            logging.info("Start validation for epoch-{}".format(ep + 1))
            acc = validate(args, model, val_loader)

            logging.info("validation time:{}".format(time() - start_time))
            if acc > best_acc:
                ckpt_path = os.path.join(args.data_path, 'ckpt', '{}-{}-{}-{}-best.pt'.format(args.model_type, args.pretrain_LM, args.lr, args.heter_embed_size))
                torch.save(model.state_dict(), ckpt_path)
                logging.info(f"Model saved to {ckpt_path}")
                best_acc = acc
                best_count = 0
            else:
                best_count += 1
                if best_count >= args.early_stop:
                    start_time = time()
                    ckpt_path = os.path.join(args.data_path, 'ckpt', '{}-{}-{}-{}-best.pt'.format(args.model_type, args.pretrain_LM, args.lr, args.heter_embed_size))
                    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                    logging.info("Start testing for best")
                    acc = validate(args, model, test_loader)
                    logging.info("test time:{}".format(time() - start_time))
                    exit()
            if args.local_rank == 0:
                torch.distributed.barrier()
        else:
            torch.distributed.barrier()

    # test
    if args.local_rank in [-1, 0]:
        start_time = time()
        # load checkpoint
        ckpt_path = os.path.join(args.data_path, 'ckpt', '{}-{}-{}-{}-best.pt'.format(args.model_type, args.pretrain_LM, args.lr, args.heter_embed_size))
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        logging.info('load ckpt:{}'.format(ckpt_path))
        acc = validate(args, model, test_loader)
        logging.info("test time:{}".format(time() - start_time))
        if args.local_rank == 0:
            torch.distributed.barrier()
    else:
        torch.distributed.barrier()
    if args.local_rank != -1:
        cleanup()


@torch.no_grad()
def validate(args, model, dataloader):
    model.eval()

    count = 0
    metrics_total = defaultdict(float)
    for step, batch in enumerate(tqdm(dataloader)):
        if args.enable_gpu:
                batch = [b.cuda() for b in batch]

        score, label = model.test(*batch)
        pred = torch.argmax(score, 1)

        if step == 0:
            preds = np.copy(pred.cpu())
            labels = np.copy(label.cpu())
            scores = np.copy(score.cpu())
        else:
            preds = np.concatenate((preds,pred.cpu()),0)
            labels = np.concatenate((labels,label.cpu()),0)
            scores = np.concatenate((scores,score.cpu()),0)

    # calculate F1 score
    metrics_total['F1_macro'] = f1_score(labels, preds, average='macro')
    metrics_total['F1_micro'] = f1_score(labels, preds, average='micro')
    metrics_total['main'] = metrics_total['F1_macro']

    logging.info("{}:{}".format('main', metrics_total['main']))
    logging.info("{}:{}".format('F1_macro', metrics_total['F1_macro']))
    logging.info("{}:{}".format('F1_micro', metrics_total['F1_micro']))

    return metrics_total['main']


def test(args):

    # define tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # load dataset
    if args.data_mode in ['bert']:
        args.user_num, args.item_num, args.class_num = pickle.load(open(os.path.join(args.data_path, 'node_num.pkl'),'rb'))
        test_set = load_dataset_bert(args, tokenizer, evaluate=True, test=True)
    else:
        raise ValueError('Data Mode is Incorrect here!')

    test_sampler = SequentialSampler(test_set) if args.local_rank == -1 else DistributedSampler(test_set)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, sampler=test_sampler)
    print('Dataset Loading Over!')

    # define model
    model = load_bert(args)
    logging.info('loading model: {}'.format(args.model_type))
    model = model.cuda()

    # load checkpoint
    start_time = time()
    checkpoint = torch.load(args.load_ckpt_name, map_location="cpu")
    model.load_state_dict(checkpoint)
    # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    logging.info('load ckpt:{}'.format(args.load_ckpt_name))

    # test
    validate(args, model, test_loader)
    logging.info("test time:{}".format(time() - start_time))
