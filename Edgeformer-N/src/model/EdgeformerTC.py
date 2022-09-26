import os
import math
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_bert import BertSelfAttention, BertLayer, BertEmbeddings, BertPreTrainedModel

from src.utils import roc_auc_score, mrr_score, ndcg_score

from IPython import embed


class GraphSingleAggregation(BertSelfAttention):
    def __init__(self, config):
        super(GraphSingleAggregation, self).__init__(config)
        self.output_attentions = False

    def forward(self, hidden_states, center_embedding, attention_mask):
        
        # stop point 5
        # embed()

        # query = self.query(hidden_states[:, :1])  # B 1 D
        query = self.query(center_embedding.unsqueeze(1)) # B 1 D
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        station_embed = self.multi_head_attention(query=query,
                                                    key=key,
                                                    value=value,
                                                    attention_mask=attention_mask)[0]  # B 1 D
        
        station_embed = station_embed.squeeze(1)

        return station_embed

    def multi_head_attention(self, query, key, value, attention_mask):
        query_layer = self.transpose_for_scores(query)
        key_layer = self.transpose_for_scores(key)
        value_layer = self.transpose_for_scores(value)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return (context_layer, attention_probs) if self.output_attentions else (context_layer,)


class GraphAggregation(BertSelfAttention):
    def __init__(self, config):
        super(GraphAggregation, self).__init__(config)
        self.output_attentions = False

    def forward(self, hidden_states, attention_mask):
        
        # stop point 5
        # embed()

        # query = self.query(hidden_states[:, :1])  # B 1 D
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        station_embed = self.multi_head_attention(query=query,
                                                    key=key,
                                                    value=value,
                                                    attention_mask=attention_mask)[0]  # B 1 D
        
        station_embed = station_embed.squeeze(1)

        return station_embed

    def multi_head_attention(self, query, key, value, attention_mask):
        query_layer = self.transpose_for_scores(query)
        key_layer = self.transpose_for_scores(key)
        value_layer = self.transpose_for_scores(value)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return (context_layer, attention_probs) if self.output_attentions else (context_layer,)


class EdgeFormerEncoderTC(nn.Module):
    def __init__(self, config):
        super(EdgeFormerEncoderTC, self).__init__()

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

        self.graph_attention = GraphAggregation(config=config)

    def forward(self,
                hidden_states,
                attention_mask,
                center_embedding,
                neighbor_embedding,
                neighbor_mask):

        all_hidden_states = ()
        all_attentions = ()

        all_nodes_num, seq_length, emb_dim = hidden_states.shape
        batch_size, _, _, _ = neighbor_mask.shape
        subgraph_node_num = neighbor_embedding.shape[1]

        # stop point 2
        # embed()

        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if i > 0:

                # stop point 4
                # embed()

                hidden_states = hidden_states.view(batch_size, subgraph_node_num, seq_length, emb_dim)  # B SN L D
                cls_emb = hidden_states[:, :, 3].clone()  # B SN D

                # input_embed = cls_emb * center_embedding.unsqueeze(1) * neighbor_embedding
                input_embed = cls_emb

                ################### You may add edge-type specific transfer here on input_embed. ######################
                # input_embed = torch.cat((W1(input_embed[:,:int(input_embed.shape[1] / 2)]),W2(input_embed[:,int(input_embed.shape[1] / 2):]), dim=1)
                ################### You may add edge-type specific transfer here on input_embed. ######################

                station_emb = self.graph_attention(hidden_states=input_embed, attention_mask=neighbor_mask)  # B D

                # update the station in the query/key
                hidden_states[:, :, 0] = center_embedding.unsqueeze(1)
                hidden_states[:, :, 1] = neighbor_embedding
                hidden_states[:, :, 2] = station_emb
                hidden_states = hidden_states.view(all_nodes_num, seq_length, emb_dim)

                layer_outputs = layer_module(hidden_states, attention_mask=attention_mask)

            else:
                # stop point 3
                # embed()

                temp_attention_mask = attention_mask.clone()
                # temp_attention_mask[::subgraph_node_num, :, :, 0] = -10000.0
                temp_attention_mask[:, :, :, :3] = -10000.0
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


class EdgeFormersTC(BertPreTrainedModel):
    def __init__(self, config):
        super(EdgeFormersTC, self).__init__(config=config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.embeddings = BertEmbeddings(config=config)
        self.encoder = EdgeFormerEncoderTC(config=config)

        self.graph_attention = GraphSingleAggregation(config=config)

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

    def forward(self,
                input_ids, 
                attention_mask, 
                center_id_batch, 
                neighbor_ids_batch, 
                neighbor_mask_batch):
        all_nodes_num, seq_length = input_ids.shape
        batch_size, _ = neighbor_mask_batch.shape
        # batch_size, subgraph_node_num = neighbor_mask.shape
        subgraph_node_num = neighbor_ids_batch.shape[1]

        # obtain embedding
        embedding_output = self.embeddings(input_ids=input_ids)
        center_node_embed = self.node_to_text_transform(self.node_embedding[center_id_batch])
        neighbor_node_embed = self.node_to_text_transform(self.node_embedding[neighbor_ids_batch])

        # Add station attention mask
        # station_mask = torch.zeros((all_nodes_num, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        station_mask = torch.ones((all_nodes_num, 3), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([station_mask, attention_mask], dim=-1)  # N 3+L
        # attention_mask[::(subgraph_node_num), 0] = 1.0  # only use the station for main nodes

        extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0
        neighbor_mask = (1.0 - neighbor_mask_batch[:, None, None, :]) * -10000.0

        # Add station_placeholder
        station_placeholder = torch.zeros(all_nodes_num, 3, embedding_output.size(-1)).type(
            embedding_output.dtype).to(embedding_output.device)
        embedding_output = torch.cat([station_placeholder, embedding_output], dim=1)  # N 3+L D

        # stop point 1
        # embed()

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            center_embedding=center_node_embed, 
            neighbor_embedding=neighbor_node_embed, 
            neighbor_mask=neighbor_mask)

        # stop point 5
        # embed()

        encoder_outputs = encoder_outputs[0][:,1].view(batch_size, subgraph_node_num, -1)
        # encoder_outputs = encoder_outputs[0][:,3].view(batch_size, subgraph_node_num, -1)
        encoder_outputs = self.graph_attention(hidden_states=encoder_outputs, center_embedding=center_node_embed, attention_mask=neighbor_mask)  # B D

        return encoder_outputs


class EdgeFormersTCForNeighborPredict(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = EdgeFormersTC(config)
        self.hidden_size = config.hidden_size
        self.init_weights()

    def init_node_embed(self, pretrain_embed, pretrain_mode, pretrain_dir):
        self.bert.init_node_embed(pretrain_embed, pretrain_mode, pretrain_dir, self.node_num, self.heter_embed_size)

    def infer(self, center_id_batch, tokens_batch, attention_mask_batch,
                neighbor_ids_batch, neighbor_mask_batch):
        '''
        B: batch size, N: 1 + neighbour_num, L: max_token_len, D: hidden dimmension
        '''

        B, N, L = tokens_batch.shape
        D = self.hidden_size
        input_ids = tokens_batch.view(B * N, L)
        attention_mask = attention_mask_batch.view(B * N, L)

        node_embeddings = self.bert(input_ids, attention_mask, center_id_batch, neighbor_ids_batch, neighbor_mask_batch)
        # last_hidden_states = hidden_states[0]
        
        # cls_embeddings = last_hidden_states[:, 1].view(B, N, D)  # [B,N,D]
        # node_embeddings = cls_embeddings[:, 0, :]  # [B,D]

        return node_embeddings

    def test(self, query_ids_batch, token_query_edges_batch, attention_query_edges_batch, query_neighbor_ids_batch, query_neighbor_mask_batch,
                key_ids_batch, token_key_edges_batch, attention_key_edges_batch, key_neighbor_ids_batch, key_neighbor_mask_batch,
                **kwargs):
        
        query_embeddings = self.infer(query_ids_batch,token_query_edges_batch, attention_query_edges_batch, 
                                        query_neighbor_ids_batch, query_neighbor_mask_batch)
        key_embeddings = self.infer(key_ids_batch,token_key_edges_batch, attention_key_edges_batch, 
                                        key_neighbor_ids_batch, key_neighbor_mask_batch)
        
        scores = torch.matmul(query_embeddings, key_embeddings.transpose(0, 1))
        labels = torch.arange(start=0, end=scores.shape[0], dtype=torch.long, device=scores.device)

        predictions = torch.argmax(scores, dim=-1)
        acc = (torch.sum((predictions == labels)) / labels.shape[0]).item()

        scores = scores.cpu().numpy()
        labels = F.one_hot(labels).cpu().numpy()
        auc_all = [roc_auc_score(labels[i], scores[i]) for i in range(labels.shape[0])]
        auc = np.mean(auc_all)
        mrr_all = [mrr_score(labels[i], scores[i]) for i in range(labels.shape[0])]
        mrr = np.mean(mrr_all)
        ndcg_all = [ndcg_score(labels[i], scores[i], labels.shape[1]) for i in range(labels.shape[0])]
        ndcg = np.mean(ndcg_all)

        return {
            "main": acc,
            "acc": acc,
            "auc": auc,
            "mrr": mrr,
            "ndcg": ndcg
        }

    def forward(self, query_ids_batch, token_query_edges_batch, attention_query_edges_batch, query_neighbor_ids_batch, query_neighbor_mask_batch,
                key_ids_batch, token_key_edges_batch, attention_key_edges_batch, key_neighbor_ids_batch, key_neighbor_mask_batch,
                **kwargs):

        query_embeddings = self.infer(query_ids_batch,token_query_edges_batch, attention_query_edges_batch, 
                                        query_neighbor_ids_batch, query_neighbor_mask_batch)
        key_embeddings = self.infer(key_ids_batch,token_key_edges_batch, attention_key_edges_batch, 
                                        key_neighbor_ids_batch, key_neighbor_mask_batch)

        score = torch.matmul(query_embeddings, key_embeddings.transpose(0, 1))
        labels = torch.arange(start=0, end=score.shape[0], dtype=torch.long, device=score.device)
        loss = F.cross_entropy(score, labels)

        return loss
