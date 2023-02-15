import os
import sys
import math
import pickle
import random
import logging
from tqdm import tqdm

import numpy as np
import torch

from torch.utils.data.dataset import Dataset, TensorDataset
from transformers import BertTokenizer, BertTokenizerFast

logger = logging.getLogger(__name__)


def load_dataset_bert(args, tokenizer, evaluate=False, test=False):
    '''
    features : (token_query_and_neighbors, attention_query_and_neighbors, mask_query_and_neighbors), (token_key_and_neighbors, attention_key_and_neighbors, mask_key_and_neighbors)
    '''
    assert args.data_mode in ['bert']

    # block for processes which are not the core process
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # load data features from cache or dataset file
    evaluation_set_name = 'test' if test else 'val'
    cached_features_file = os.path.join(args.data_path, 'cached_{}_{}_{}_{}'.format(
        args.data_mode,
        evaluation_set_name if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_length)))
    
    # exist or not
    if os.path.exists(cached_features_file):
        if args.local_rank in [-1, 0]:
            logger.info(f"Loading features from cached file {cached_features_file}")
        features = pickle.load(open(cached_features_file,'rb'))
    else:
        if args.local_rank in [-1, 0]:
            logger.info("Creating features from dataset file at %s",
                    args.data_path)

        read_file = (evaluation_set_name if evaluate else 'train') + f'.tsv'

        # assert args.max_length == 512

        features = read_process_data_bert(os.path.join(args.data_path, read_file), tokenizer, args.max_length)
        logger.info(f"Saving features into cached file {cached_features_file}")
        pickle.dump(features, open(cached_features_file, 'wb'))

    if args.local_rank == 0:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # convert to Tensors and build dataset
    token_query_edges = torch.LongTensor(features[0][0])
    attention_query_edges = torch.LongTensor(features[0][1])
    
    query_node = torch.LongTensor(features[1])
    key_node = torch.LongTensor(features[2]) + args.user_num
    edge_labels_id = torch.LongTensor(features[3])
    edge_labels = torch.zeros(query_node.shape[0], args.class_num).scatter_(1, edge_labels_id.unsqueeze(-1), 1)

    ################################### Be Careful about the Reindex Here. ###################################
    dataset = TensorDataset(token_query_edges, attention_query_edges, query_node, key_node, edge_labels)

    return dataset


def read_process_data_bert(dir, tokenizer, max_length):
    '''
    Each line is a user/item node pair. Each node is made up of [itself, pos_neighbour * positive reviews, neg_neighbour * negative reviews, pos_neighbour * positive neighbor, neg_neighbour * negative neighbor, edgeType].
    Edge type in the sampled ego-graph is retricted by position in the tensor.
    '''
    token_edges = []
    attention_edges = []

    edge_labels = []
    query_node_idx = []
    key_node_idx = []

    with open(dir) as f:
        data = f.readlines()
        for line in tqdm(data):
            a = line.strip().split('\$\$')
            if len(a) == 4:
                text, query_n, key_n, label = a
            else:
                print(len(a))
                raise ValueError('stop')

            encoded_text = tokenizer.batch_encode_plus([text], max_length=max_length, padding='max_length', truncation=True)

            token_edges.append(encoded_text['input_ids'])
            attention_edges.append(encoded_text['attention_mask'])
            edge_labels.append(int(label))
            query_node_idx.append(int(query_n))
            key_node_idx.append(int(key_n))

    return (token_edges, attention_edges), query_node_idx, key_node_idx, edge_labels
