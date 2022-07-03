from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from .utils import my_con_loss

from .att_model import pack_wrapper, AttModel


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def memory_querying_responding(query, key, value, mask=None, dropout=None, topk=32):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    selected_scores, idx = scores.topk(topk)
    # query [8, 8, 49/98, 64]) value [8, 8, 2048, 64] score [8, 8, 49/98, 2048]
    # idx 8x8x98x32
    dummy_value = value.unsqueeze(2).expand(idx.size(0), idx.size(1), idx.size(2), value.size(-2), value.size(-1)) #[8, 8, 98, 2048, 64]
    dummy_idx = idx.unsqueeze(-1).expand(idx.size(0), idx.size(1), idx.size(2), idx.size(3), value.size(-1)) # [8, 8, 98, 32, 64]

    selected_value = torch.gather(dummy_value, 3, dummy_idx) # [8, 8, 98, 32, 64]

    p_attn = F.softmax(selected_scores, dim=-1)  # [8, 8, 98, 32]

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn.unsqueeze(3), selected_value).squeeze(3), p_attn


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, cmn):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.cmn = cmn

        self.fuse_feature = nn.Linear(512*2, 512)

    def forward(self, src, tgt, src_mask, tgt_mask, memory_matrix, cmn_masks = None, labels = None):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask, memory_matrix=memory_matrix,
                           cmn_masks = cmn_masks, labels = labels)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask) # [12,98,512]

    def decode(self, memory, src_mask, tgt, tgt_mask, past=None, memory_matrix=None, cmn_masks = None, labels = None):
        embeddings = self.tgt_embed(tgt)

        cmn_masks = cmn_masks.unsqueeze(1).expand(cmn_masks.shape[0], embeddings.size(1), cmn_masks.shape[-1])

        # Cross-modal prototype querying and responding for textual features

        responses = self.cmn(embeddings, memory_matrix, memory_matrix, cmn_masks)

        #embeddings = embeddings + responses
        embeddings = self.fuse_feature(torch.cat((embeddings, responses), dim=2))

        return self.decoder(embeddings, memory, src_mask, tgt_mask, past=past), embeddings, responses


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        _x = sublayer(self.norm(x))
        if type(_x) is tuple:
            return x + self.dropout(_x[0]), _x[1]
        return x + self.dropout(_x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask, past=None):
        if past is not None:
            present = [[], []]
            x = x[:, -1:]
            tgt_mask = tgt_mask[:, -1:] if tgt_mask is not None else None
            past = list(zip(past[0].split(2, dim=0), past[1].split(2, dim=0)))
        else:
            past = [None] * len(self.layers)
        for i, (layer, layer_past) in enumerate(zip(self.layers, past)):
            x = layer(x, memory, src_mask, tgt_mask,
                      layer_past)
            if layer_past is not None:
                present[0].append(x[1][0])
                present[1].append(x[1][1])
                x = x[0]
        if past[0] is None:
            return self.norm(x)
        else:
            return self.norm(x), [torch.cat(present[0], 0), torch.cat(present[1], 0)]


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask, layer_past=None):
        m = memory
        if layer_past is None:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
            return self.sublayer[2](x, self.feed_forward)
        else:
            present = [None, None]
            x, present[0] = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, layer_past[0]))
            x, present[1] = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, layer_past[1]))
            return self.sublayer[2](x, self.feed_forward), present


class MultiThreadMemory(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, topk=32):
        super(MultiThreadMemory, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.topk = topk

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]

        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])
        
        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        x, self.attn = memory_querying_responding(query, key, value, mask=mask, dropout=self.dropout, topk=self.topk)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]

        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2) # origin: batchsize x sequence_num x head_num x head_dim
             for x in [query, key, value]]

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        self.temp=vocab

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class EncoderDecoder(AttModel):

    def make_model(self, tgt_vocab, cmn):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout), self.num_layers),
            nn.Sequential(c(position)),
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)), cmn
        )
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer, mode = 'train'):
        super(EncoderDecoder, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.topk = args.topk
        self.num_cluster = args.num_cluster
        self.img_margin = args.img_con_margin
        self.txt_margin = args.txt_con_margin
        self.num_protype = args.num_protype

        self.img_cls_head = nn.Sequential(nn.Linear(args.cmm_dim, args.cmm_dim), nn.Linear(args.cmm_dim, 14))

        self.txt_cls_head = nn.Sequential(nn.Linear(args.cmm_dim, args.cmm_dim), nn.Linear(args.cmm_dim, 14))

        self.txt_dim_reduction = nn.Linear(args.d_txt_ebd, args.cmm_dim)

        self.dim_reduction = nn.Linear(args.d_txt_ebd + args.d_img_ebd, args.cmm_dim)

        self.img_dim_reduction = nn.Linear(args.d_img_ebd, args.cmm_dim)

        self.fuse_feature = nn.Linear(args.d_model*2, args.d_model)


        tgt_vocab = self.vocab_size + 1


        self.cmn = MultiThreadMemory(args.num_heads, args.d_model, topk=args.topk)

        self.model = self.make_model(tgt_vocab, self.cmn)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

        self.protypes = nn.Parameter(torch.FloatTensor(args.num_protype * args.num_cluster, args.d_txt_ebd + args.d_img_ebd))
        if mode == 'train':
            init_protypes = torch.load(args.init_protypes_path).float()
            self.protypes = nn.Parameter(init_protypes)



    def init_hidden(self, bsz):
        return []


    def _prepare_feature(self, fc_feats, att_feats, att_masks, labels = None):
        att_feats, seq, att_masks, seq_mask, query_matrix, cmn_masks, _ = \
            self._prepare_feature_forward(att_feats, att_masks, labels=labels)
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks, labels, query_matrix, cmn_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None, labels=None,):

        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)

        max_num_protype = max(labels.sum(-1)) * self.num_protype
        protypes = self.dim_reduction(self.protypes)
        query_matrix = protypes.new_zeros(att_feats.size(0), max_num_protype.int(), protypes.shape[-1])
        cmn_masks = protypes.new_zeros(query_matrix.shape[0], att_feats.size(1), max_num_protype.int())

        for i in range(att_feats.size(0)):
            cur_query_matrix = []
            for j in range(len(labels[i])):
                if labels[i, j] == 1:
                        cur_query_matrix.extend(
                            protypes[j*self.num_protype:(j+1)*self.num_protype, :])

            cur_query_matrix = torch.stack(cur_query_matrix, 0)

            query_matrix[i, :cur_query_matrix.shape[0], :] = cur_query_matrix
            cmn_masks[i, :, :cur_query_matrix.shape[0]] = 1


        responses = self.cmn(att_feats, query_matrix, query_matrix, cmn_masks)

        # feature interaction
        att_feats = self.fuse_feature(torch.cat((att_feats, responses), dim=2))


        att_masks = att_masks.unsqueeze(-2)
        if seq is not None:
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask, query_matrix, cmn_masks[:,0,:], responses

    def _forward(self, fc_feats, att_feats, seq, att_masks=None, labels=None):
        att_feats, seq, att_masks, seq_mask, query_matrix, cmn_masks, img_responses = \
            self._prepare_feature_forward(att_feats, att_masks, seq, labels)
        out, txt_feats, txt_responses = self.model(att_feats, seq, att_masks, seq_mask, memory_matrix=query_matrix,
                         cmn_masks = cmn_masks, labels = labels)
        outputs = F.log_softmax(self.logit(out), dim=-1)
        img_con_loss = my_con_loss(torch.mean(img_responses, dim=1), num_classes= self.num_cluster,
                               num_protypes = self.num_protype, labels = labels, margin = self.img_margin)
        img_con_loss = img_con_loss.unsqueeze(0)  # for  multi-gpu setting
        txt_con_loss = my_con_loss(torch.mean(txt_responses, dim=1), num_classes= self.num_cluster,
                               num_protypes = self.num_protype, labels = labels, margin = self.txt_margin)
        txt_con_loss = txt_con_loss.unsqueeze(0)  # for  multi-gpu setting

        img_bce_loss = self.img_cls_head(torch.mean(att_feats, dim=1))
        txt_bce_loss = self.txt_cls_head(torch.mean(txt_feats, dim=1))

        return outputs, img_con_loss, txt_con_loss, img_bce_loss, txt_bce_loss

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask, query_matrix, cmn_masks, labels=None):
        if len(state) == 0:
            ys = it.unsqueeze(1)

            past = [fc_feats_ph.new_zeros(self.num_layers * 2, fc_feats_ph.shape[0], 0, self.d_model),
                    fc_feats_ph.new_zeros(self.num_layers * 2, fc_feats_ph.shape[0], 0, self.d_model)]
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
            past = state[1:]

        [out, past], _, _ = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device), past=past,
                                      memory_matrix=query_matrix, cmn_masks = cmn_masks, labels = labels)

        return out[:, -1], [ys.unsqueeze(0)] + past
    
def con_loss(features, labels):
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - 0.4
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss