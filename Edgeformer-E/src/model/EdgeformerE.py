import os
import math
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_bert import BertSelfAttention, BertLayer, BertEmbeddings, BertPreTrainedModel

from src.utils import roc_auc_score, mrr_score, ndcg_score

class EdgeFormerEncoderE(nn.Module):
    def __init__(self, config):
        super(EdgeFormerEncoderE, self).__init__()

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self,
                hidden_states,
                attention_mask,
                query_embedding,
                key_embedding):

        all_hidden_states = ()
        all_attentions = ()

        all_nodes_num, seq_length, emb_dim = hidden_states.shape
        # subgraph_node_num = neighbor_embedding.shape[1]

        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if i > 0:

                ################### You may add edge-type specific transfer here on input_embed. ######################
                # input_embed = torch.cat((W1(input_embed[:,:int(input_embed.shape[1] / 2)]),W2(input_embed[:,int(input_embed.shape[1] / 2):]), dim=1)
                ################### You may add edge-type specific transfer here on input_embed. ######################

                # update the station in the query/key
                hidden_states[:, 0] = query_embedding
                hidden_states[:, 1] = key_embedding

                layer_outputs = layer_module(hidden_states, attention_mask=attention_mask)

            else:

                temp_attention_mask = attention_mask.clone()
                # temp_attention_mask[::subgraph_node_num, :, :, 0] = -10000.0
                temp_attention_mask[:, :, :, :2] = -10000.0
                layer_outputs = layer_module(hidden_states, attention_mask=temp_attention_mask)

            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class EdgeFormersE(BertPreTrainedModel):
    def __init__(self, config):
        super(EdgeFormersE, self).__init__(config=config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.embeddings = BertEmbeddings(config=config)
        self.encoder = EdgeFormerEncoderE(config=config)

    def init_node_embed(self, pretrain_embed, pretrain_mode, pretrain_dir, node_num, heter_embed_size):
        self.node_num = node_num
        self.heter_embed_size = heter_embed_size

        if not pretrain_embed:
            self.node_embedding = nn.Parameter(torch.FloatTensor(self.node_num, self.heter_embed_size))
            nn.init.xavier_normal_(self.node_embedding)
            self.node_to_text_transform = nn.Linear(self.heter_embed_size, self.hidden_size)

        else:
            if pretrain_mode == 'BERTMF':
                checkpoint = pickle.load(open(os.path.join(pretrain_dir, f'BERTMF_{heter_embed_size}.pt'),'rb'))
                self.node_embedding = nn.Parameter(checkpoint['author_embeddings']) # this "author_embedding" name might be ambiguous, but it's not a bug, relax
                self.node_to_text_transform = nn.Linear(self.heter_embed_size, self.hidden_size)
                with torch.no_grad():
                    self.node_to_text_transform.weight.copy_(checkpoint['linear.weight'])
                    self.node_to_text_transform.bias.copy_(checkpoint['linear.bias'])
            elif pretrain_mode == 'MF':
                checkpoint = torch.load(os.path.join(pretrain_dir, f'MF_{heter_embed_size}.pt'), map_location='cpu')
                self.node_embedding = nn.Parameter(checkpoint['node_embedding'])
                self.node_to_text_transform = nn.Linear(self.heter_embed_size, self.hidden_size)
            else:
                raise ValueError('Wrong pretrain mode!')

    def forward(self, input_ids, attention_mask, query_node_idx, key_node_idx):

        all_nodes_num, seq_length = input_ids.shape
        # batch_size, _ = neighbor_mask_batch.shape
        # # batch_size, subgraph_node_num = neighbor_mask.shape
        # subgraph_node_num = neighbor_ids_batch.shape[1]

        # obtain embedding
        embedding_output = self.embeddings(input_ids=input_ids)
        query_node_embed = self.node_to_text_transform(self.node_embedding[query_node_idx])
        key_node_embed = self.node_to_text_transform(self.node_embedding[key_node_idx])

        # Add station attention mask
        station_mask = torch.ones((all_nodes_num, 2), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([station_mask, attention_mask], dim=-1)  # N 2+L
        # attention_mask[::(subgraph_node_num), 0] = 1.0  # only use the station for main nodes

        extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0

        # Add station_placeholder
        station_placeholder = torch.zeros(all_nodes_num, 2, embedding_output.size(-1)).type(
            embedding_output.dtype).to(embedding_output.device)
        embedding_output = torch.cat([station_placeholder, embedding_output], dim=1)  # N 2+L D

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            query_embedding=query_node_embed, 
            key_embedding=key_node_embed)

        # encoder_outputs = encoder_outputs[0][:,1].view(batch_size, subgraph_node_num, -1)
        encoder_outputs = encoder_outputs[0][:,2]

        return encoder_outputs


class EdgeFormersForEdgeClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = EdgeFormersE(config)
        self.hidden_size = config.hidden_size
        self.init_weights()
        self.classifier = nn.Sequential(nn.Linear(config.hidden_size, config.class_num),
                                    nn.Softmax(dim=1))
        self.loss_func = nn.BCELoss()

    def init_node_embed(self, pretrain_embed, pretrain_mode, pretrain_dir):
        self.bert.init_node_embed(pretrain_embed, pretrain_mode, pretrain_dir, self.node_num, self.heter_embed_size)

    def infer(self, token_edges_batch, attention_edges_batch, query_node_idx, key_node_idx):

        edge_embeddings = self.bert(token_edges_batch.squeeze(1), attention_edges_batch.squeeze(1), query_node_idx, key_node_idx)

        return edge_embeddings

    def test(self, token_edges_batch, attention_edges_batch, query_node_idx, key_node_idx, edge_labels, **kwargs):
        
        edge_embeddings = self.infer(token_edges_batch, attention_edges_batch, query_node_idx, key_node_idx)
        scores = self.classifier(edge_embeddings)

        label_id = torch.argmax(edge_labels, 1)

        return scores, label_id

    def forward(self, token_edges_batch, attention_edges_batch, query_node_idx, key_node_idx, edge_labels, **kwargs):
    
        edge_embeddings = self.infer(token_edges_batch, attention_edges_batch, query_node_idx, key_node_idx)
        logit = self.classifier(edge_embeddings)

        loss = self.loss_func(logit, edge_labels)

        return loss
