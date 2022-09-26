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

from IPython import embed

logger = logging.getLogger(__name__)


def load_dataset_text(args, tokenizer, evaluate=False, test=False):
    '''
    features : (token_query_and_neighbors, attention_query_and_neighbors, mask_query_and_neighbors), (token_key_and_neighbors, attention_key_and_neighbors, mask_key_and_neighbors)
    '''
    assert args.data_mode in ['text']

    # block for processes which are not the core process
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # load data features from cache or dataset file
    evaluation_set_name = 'test' if test else 'val'
    cached_features_file = os.path.join(args.data_path, 'cached_{}_{}_{}_{}_{}_{}_{}_{}'.format(
        args.data_mode,
        evaluation_set_name if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_length),
        args.user_pos_neighbour,
        args.user_neg_neighbour, 
        args.item_pos_neighbour, 
        args.item_neg_neighbour))
    
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

        features = read_process_data_text(os.path.join(args.data_path, read_file), tokenizer, args.max_length, args.user_pos_neighbour, args.user_neg_neighbour, args.item_pos_neighbour, args.item_neg_neighbour)
        logger.info(f"Saving features into cached file {cached_features_file}")
        pickle.dump(features, open(cached_features_file, 'wb'))

    if args.local_rank == 0:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # convert to Tensors and build dataset
    token_query_edges = torch.LongTensor(features[0][0])
    attention_query_edges = torch.LongTensor(features[0][1])
    query_neighbor_ids = torch.LongTensor(features[0][2])
    query_neighbor_mask = torch.LongTensor(features[0][3])
    
    token_key_edges = torch.LongTensor(features[1][0])
    attention_key_edges = torch.LongTensor(features[1][1])
    key_neighbor_ids = torch.LongTensor(features[1][2])
    key_neighbor_mask = torch.LongTensor(features[1][3])

    query_ids = torch.LongTensor(features[2])
    key_ids = torch.LongTensor(features[3])
    edge_labels = torch.LongTensor(features[4])

    assert token_query_edges.shape[1] == args.user_pos_neighbour + args.user_neg_neighbour
    assert attention_query_edges.shape[1] == args.user_pos_neighbour + args.user_neg_neighbour
    assert query_neighbor_ids.shape[1] == args.user_pos_neighbour + args.user_neg_neighbour
    assert query_neighbor_mask.shape[1] == args.user_pos_neighbour + args.user_neg_neighbour

    # embed()
    # raise ValueError('stop')

    ################################### Be Careful about the Reindex Here. ###################################
    pos_index = (edge_labels == 1)
    # dataset = TensorDataset(query_ids[pos_index], key_ids[pos_index] + args.user_num,
    #                         token_query_edges[pos_index], attention_query_edges[pos_index], query_neighbor_ids[pos_index] + args.user_num, query_neighbor_mask[pos_index],
    #                         token_key_edges[pos_index], attention_key_edges[pos_index], key_neighbor_ids[pos_index], key_neighbor_mask[pos_index])

    dataset = TensorDataset(query_ids[pos_index], token_query_edges[pos_index], attention_query_edges[pos_index], query_neighbor_ids[pos_index] + args.user_num, query_neighbor_mask[pos_index],
                            key_ids[pos_index] + args.user_num, token_key_edges[pos_index], attention_key_edges[pos_index], key_neighbor_ids[pos_index], key_neighbor_mask[pos_index])

    return dataset


def read_process_data_text(dir, tokenizer, max_length, user_pos_neighbour, user_neg_neighbour, item_pos_neighbour, item_neg_neighbour):
    '''
    Each line is a user/item node pair. Each node is made up of [itself, pos_neighbour * positive reviews, neg_neighbour * negative reviews, pos_neighbour * positive neighbor, neg_neighbour * negative neighbor, edgeType].
    Edge type in the sampled ego-graph is retricted by position in the tensor.
    '''
    token_query_edges = []
    token_key_edges = []
    attention_query_edges = []
    attention_key_edges = []

    query_neighbor_ids = []
    key_neighbor_ids = []

    query_neighbor_mask = []
    key_neighbor_mask = []

    query_ids = []
    key_ids = []
    edge_labels = []

    with open(dir) as f:
        data = f.readlines()
        for line in tqdm(data):
            a = line.strip().split('\$\$')
            if len(a) == 3:
                query_all, key_all, label = a
            else:
                print(len(a))
                # print(a)
                raise ValueError('stop')
            
            query_all = query_all.split('\*\*')
            key_all = key_all.split('\*\*')
            tmp_mask_query = []
            tmp_mask_key = []

            # make sure that length is 5 for query and key
            assert len(query_all) == 5
            assert len(key_all) == 5

            qid = query_all[0]
            query_pos_text, query_neg_text, query_pos_neighbors, query_neg_neighbors = [qq.split('\t') for qq in query_all[1:]]
            kid = key_all[0]
            key_pos_text, key_neg_text, key_pos_neighbors, key_neg_neighbors = [kk.split('\t') for kk in key_all[1:]]

            assert len(query_pos_text) == user_pos_neighbour
            assert len(query_pos_neighbors) == user_pos_neighbour
            assert len(key_pos_text) == item_pos_neighbour
            assert len(key_pos_neighbors) == item_pos_neighbour

            if user_neg_neighbour != 0:
                assert len(query_neg_text) == user_neg_neighbour
                assert len(query_neg_neighbors) == user_neg_neighbour
                assert len(key_neg_text) == item_neg_neighbour
                assert len(key_neg_neighbors) == item_neg_neighbour
            else:
                query_neg_text = []
                query_neg_neighbors = []
                key_neg_text = []
                key_neg_neighbors = []

            # split the neighbours
            query_pos_neighbors = [int(v) for v in query_pos_neighbors]
            query_neg_neighbors = [int(v) for v in query_neg_neighbors]
            qeury_neighbors = query_pos_neighbors + query_neg_neighbors

            key_pos_neighbors = [int(v) for v in key_pos_neighbors]
            key_neg_neighbors = [int(v) for v in key_neg_neighbors]
            key_neighbors = key_pos_neighbors + key_neg_neighbors

            query_neighbors_edges_text = query_pos_text + query_neg_text
            key_neighbors_edges_text = key_pos_text + key_neg_text

            # construct neighbour mask
            for v in qeury_neighbors:
                if v != -1:
                    tmp_mask_query.append(1)
                else:
                    tmp_mask_query.append(0)

            for v in key_neighbors:
                if v != -1:
                    tmp_mask_key.append(1)
                else:
                    tmp_mask_key.append(0)

            query_neighbor_ids.append(qeury_neighbors)
            key_neighbor_ids.append(key_neighbors)
            query_neighbor_mask.append(tmp_mask_query)
            key_neighbor_mask.append(tmp_mask_key)
            
            query_ids.append(int(qid))
            key_ids.append(int(kid))
            edge_labels.append(int(label))

            encoded_query_edges = tokenizer.batch_encode_plus(query_neighbors_edges_text, max_length=max_length, padding='max_length', truncation=True)
            encoded_key_edges = tokenizer.batch_encode_plus(key_neighbors_edges_text, max_length=max_length, padding='max_length', truncation=True)

            token_query_edges.append(encoded_query_edges['input_ids'])
            token_key_edges.append(encoded_key_edges['input_ids'])
            attention_query_edges.append(encoded_query_edges['attention_mask'])
            attention_key_edges.append(encoded_key_edges['attention_mask'])

    return (token_query_edges, attention_query_edges, query_neighbor_ids, query_neighbor_mask), \
        (token_key_edges, attention_key_edges, key_neighbor_ids, key_neighbor_mask), query_ids, key_ids, edge_labels
