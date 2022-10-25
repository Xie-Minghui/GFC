import torch
import torch.nn as nn
import math
from transformers import AutoModel
from transformers import RobertaModel, BertModel


class GCF(nn.Module):
    def __init__(self, args, ent2id, rel2id, triples):
        super().__init__()
        self.args = args
        self.num_steps = 2
        num_relations = len(rel2id)

        Tsize = len(triples)
        Esize = len(ent2id)
        idx = torch.LongTensor([i for i in range(Tsize)])
        self.Msubj = torch.sparse.FloatTensor(
            torch.stack((idx, triples[:,0])), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, Esize]))
        self.Mobj = torch.sparse.FloatTensor(
            torch.stack((idx, triples[:,2])), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, Esize]))
        self.Mrel = torch.sparse.FloatTensor(
            torch.stack((idx, triples[:,1])), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, num_relations]))
        print('triple size: {}'.format(Tsize))
        try:
            if args.bert_name == "bert-base-uncased":
                self.bert_encoder = BertModel.from_pretrained('/YourPath/bert-base-uncased')
            elif args.bert_name == "roberta-base":
                self.bert_encoder = RobertaModel.from_pretrained('/YourPath/roberta-base')
            else:
                raise ValueError("please input the right name of pretrained model")
        except ValueError as e:
            raise e
        dim_hidden = self.bert_encoder.config.hidden_size
        self.rel_classifier = nn.Linear(dim_hidden, num_relations)
        self.key_layer = nn.Linear(dim_hidden, dim_hidden)
        self.hop_att_layer = nn.Sequential(
            nn.Linear(dim_hidden, 1)
            # nn.Tanh()
        )

        self.high_way = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.Sigmoid()
        )

    def follow(self, e, r):
        x = torch.sparse.mm(self.Msubj, e.t()) * torch.sparse.mm(self.Mrel, r.t())
        return torch.sparse.mm(self.Mobj.t(), x).t() # [bsz, Esize]

    def forward(self, heads, questions, answers=None, entity_range=None):
        q = self.bert_encoder(**questions)
        q_embeddings, q_word_h = q.pooler_output, q.last_hidden_state # (bsz, dim_h), (bsz, len, dim_h)

        device = heads.device
        last_e = heads
        word_attns = []
        rel_probs = []
        ent_probs = []
        ctx_h_list = []
        q_word_h_hop = q_word_h
        q_word_h_dist_ctx = [0]
        for t in range(self.num_steps):
            h_key = self.key_layer(q_word_h_hop)  # [bsz, max_q, dim_h]
            q_logits = torch.matmul(h_key, q_word_h.transpose(-1, -2)) # [bsz, max_q, dim_h] * [bsz, dim_h, max_q] = [bsz, max_q, max_q]
            q_logits = q_logits.transpose(-1, -2)

            q_dist = torch.softmax(q_logits, 2)  # [bsz, max_q, max_q] 
            q_dist = q_dist * questions['attention_mask'].float().unsqueeze(1)  # [bsz, max_q, max_q]*[bsz, max_q]
            q_dist = q_dist / (torch.sum(q_dist, dim=2, keepdim=True) + 1e-6) # [bsz, max_q, max_q] 
            hop_ctx = torch.matmul(q_dist, q_word_h_hop) 
            if t == 0:
                z = 0
            else:
                z = self.high_way(q_word_h_dist_ctx[-1]) 
            if t == 0:
                q_word_h_hop = q_word_h + hop_ctx
            else:
                q_word_h_hop = q_word_h + hop_ctx + z*q_word_h_dist_ctx[-1]# [bsz, max_q, max_q]*[bsz, max_q, dim_h] = [bsz, max_q, dim_h]
            q_word_h_dist_ctx.append(hop_ctx + z*q_word_h_dist_ctx[-1])

            q_word_att = torch.sum(q_dist, dim=1, keepdim=True)  # [bsz, 1, max_q]  # 2改为1
            q_word_att = torch.softmax(q_word_att, 2)
            q_word_att = q_word_att * questions['attention_mask'].float().unsqueeze(1)  # [bsz, 1, max_q]*[bsz, max_q]
            q_word_att = q_word_att / (torch.sum(q_word_att, dim=2, keepdim=True) + 1e-6)  # [bsz, max_q, max_q]
            word_attns.append(q_word_att)  # bsz,1,q_max
            ctx_h = (q_word_h_hop.transpose(-1,-2) @ q_word_att.transpose(-1,-2)).squeeze(2)  # [bsz, dim_h, max_q] * [bsz, max_q,1]

            ctx_h_list.append(ctx_h)

            rel_logit = self.rel_classifier(ctx_h) # [bsz, num_relations]
            rel_dist = torch.sigmoid(rel_logit)
            rel_probs.append(rel_dist)

            last_e = self.follow(last_e, rel_dist)  # faster than index_add

            # reshape >1 scores to 1 in a differentiable way
            m = last_e.gt(1).float() 
            z = (m * last_e + (1-m)).detach()
            last_e = last_e / z
            ent_probs.append(last_e)

        hop_res = torch.stack(ent_probs, dim=1) # [bsz, num_hop, num_ent]

        ctx_h_history = torch.stack(ctx_h_list, dim=2)  # [bsz, dim_h, num_hop]
        hop_logit = self.hop_att_layer(ctx_h_history.transpose(-1, -2))  # bsz, num_hop, 1
        hop_attn = torch.softmax(hop_logit.transpose(-1, -2), 2).transpose(-1, -2)  # bsz, num_hop, 1

        last_e = torch.sum(hop_res * hop_attn, dim=1) # [bsz, num_ent]

        if not self.training:
            return {
                'e_score': last_e,
                'word_attns': word_attns,
                'rel_probs': rel_probs,
                'ent_probs': ent_probs,
                'hop_attn': hop_attn.squeeze(2)
            }
        else:
            weight = answers * 9 + 1 
            loss = torch.sum(entity_range * weight * torch.pow(last_e - answers, 2)) / torch.sum(entity_range * weight)

            return {'loss': loss}
