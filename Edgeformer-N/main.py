import os
import logging
import argparse
from pathlib import Path
import torch.multiprocessing as mp

from src.run import train, test, infer
from src.utils import setuplogging, str2bool, set_seed

parser = argparse.ArgumentParser(description='Study for Edge Text-Rich Networks.')
parser.add_argument("--mode", type=str, default="train", choices=['train', 'test', 'infer'])
parser.add_argument("--data_path", type=str, default="movie/", choices=['movie/', 'Apps/', 'children/', 'crime_book/', 'stackoverflow/'])
# parser.add_argument("--model_dir", type=str, default='ckpt/', choices=['ckpt/', 'ckpt-test/'])  # path to save
parser.add_argument("--data_mode", default="text", type=str, choices=['text'])
parser.add_argument("--pretrain_embed", type=str2bool, default=False) # use pretrain node embedding or not
parser.add_argument("--pretrain_dir", default="movie/pretrain", type=str, choices=['movie/pretrain']) # pretrain node embedding dir
parser.add_argument("--pretrain_mode", default="MF", type=str, choices=['MF','BERTMF']) # pretrain node embedding dir

# turing
parser.add_argument("--model_type", default="EdgeformerN", type=str, choices=['EdgeformerN'])
parser.add_argument("--pretrain_LM", type=str2bool, default=True)
parser.add_argument("--heter_embed_size", type=int, default=64)

# some parameters fixed depend on dataset
parser.add_argument("--max_length", type=int, default=64) # this parameter should be the same for all models for one particular dataset
parser.add_argument("--train_batch_size", type=int, default=30)
parser.add_argument("--val_batch_size", type=int, default=100)
parser.add_argument("--test_batch_size", type=int, default=100)
parser.add_argument("--warmup_steps", type=int, default=1000)

# distribution
parser.add_argument("--local_rank", type=int, default=-1)

# model training (these parameters can be fixed)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--epochs", type=int, default=100) # 12 for running with scheduler
parser.add_argument("--early_stop", type=int, default=3)
parser.add_argument("--log_steps", type=int, default=100)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--load", type=str2bool, default=False)
parser.add_argument("--max_grad_norm", type=int, default=1)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--adam_epsilon", type=float, default=1e-8)
parser.add_argument("--enable_gpu", type=str2bool, default=True)

# load checkpoint or test
parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str,
                    help="Path to pre-trained model or shortcut name. ")
parser.add_argument(
        "--load_ckpt_name",
        type=str,
        help="choose which ckpt to load and test"
    )

# half float
parser.add_argument("--fp16", type=str2bool, default=True)

args = parser.parse_args()

# # pretrain dir
# if args.data_path in ['movie/', 'movie/debug']:
#     args.pretrain_dir = 'movie/pretrain'
# else:
#     raise ValueError('stop')

# model data mode
if args.model_type in ['EdgeformerN']:
    assert args.data_mode == 'text'
else:
    raise ValueError('Wrong Data Mode!')


if args.local_rank in [-1,0]:
    logging.info(args)
    print(args)


if __name__ == "__main__":

    set_seed(args.random_seed)
    setuplogging()

    if args.local_rank in [-1,0]:
        print(os.getcwd())
    # Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == 'train':
        if args.local_rank in [-1,0]:
            print('-----------train------------')
        train(args)

    if args.mode == 'test':
        print('-------------test--------------')
        ################## You should use single GPU for testing. ####################
        assert args.local_rank == -1
        test(args)

    if args.mode == 'infer':
        print('-------------infer--------------')
        ################## You should use single GPU for infering. ####################
        assert args.local_rank == -1
        infer(args)
