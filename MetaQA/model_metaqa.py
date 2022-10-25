import torch
import torch.nn as nn
import math

from utils.BiGRU import GRU, BiGRU
from .Knowledge_graph import KnowledgeGraph

class GCF(nn.Module):
    def __init__(self, args, dim_word, dim_hidden, vocab):
        super().__init__()
        self.args = args
        self.vocab = vocab
        self.kg = KnowledgeGraph(args, vocab)
        num_words = len(vocab['word2id'])
        num_entities = len(vocab['entity2id'])
        num_relations = len(vocab['relation2id'])
        self.num_steps = args.num_steps
        self.aux_hop = args.aux_hop

        self.question_encoder = BiGRU(dim_word, dim_hidden, num_layers=1, dropout=0.2)

        self.word_embeddings = nn.Embedding(num_words, dim_word)
        self.word_dropout = nn.Dropout(0.2)
        self.step_encoders = []
        for i in range(self.num_steps):
            m = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
            )
            self.step_encoders.append(m)
            self.add_module('step_encoders_{}'.format(i), m)
        self.key_layer = nn.Linear(dim_hidden, dim_hidden)
        self.rel_classifier = nn.Linear(dim_hidden, num_relations)
        self.hop_att_layer = nn.Sequential(
            nn.Linear(dim_hidden, 1)
        )
        self.high_way = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.Sigmoid()
        )

    def follow(self, e, r):
        x = torch.sparse.mm(self.kg.Msubj, e.t()) * torch.sparse.mm(self.kg.Mrel, r.t())
        return torch.sparse.mm(self.kg.Mobj.t(), x).t()  # [bsz, Esize]

    def sequence_mask(self, lengths, max_len=16):
        batch_size = lengths.numel()
        # max_len = max_len or lengths.max()
        return (torch.arange(0, max_len, device=lengths.device)
                .type_as(lengths)
                .unsqueeze(0).expand(batch_size, max_len)
                .lt(lengths.unsqueeze(1)))

    def forward(self, questions, e_s, answers=None, hop=None):
        question_lens = questions.size(1) - questions.eq(0).long().sum(dim=1)  # 0 means <PAD>
        q_word_emb = self.word_dropout(self.word_embeddings(questions))  # [bsz, max_q, dim_hidden]
        q_word_h, q_embeddings, q_hn = self.question_encoder(q_word_emb,
                                                             question_lens)  # [bsz, max_q, dim_h], [bsz, dim_h], [num_layers, bsz, dim_h]

        device = q_word_h.device
        bsz = q_word_h.size(0)
        dim_h = q_word_h.size(-1)
        last_e = e_s
        word_attns = []
        rel_probs = []
        ent_probs = []
        ctx_h_list = []
        q_word_h_hop = q_word_h
        q_word_h_dist_ctx = [0]
        att_mask = self.sequence_mask(question_lens)
        for t in range(self.num_steps):
            h_key = self.step_encoders[t](q_word_h_hop)
            q_logits = torch.matmul(h_key, q_word_h.transpose(-1,
                                                              -2))  # [bsz, max_q, dim_h] * [bsz, dim_h, max_q] = [bsz, max_q, max_q]
            q_logits = q_logits.transpose(-1, -2) 

            q_dist = torch.softmax(q_logits, 2)  # [bsz, max_q, max_q] 
            q_dist = q_dist * att_mask.float().unsqueeze(1)
            q_dist = q_dist / (torch.sum(q_dist, dim=2, keepdim=True) + 1e-6)  # [bsz, max_q, max_q]
            hop_ctx = torch.matmul(q_dist, q_word_h_hop) 
            if t == 0:
                z = 0
            else:
                z = self.high_way(q_word_h_dist_ctx[-1])
            if t == 0:
                q_word_h_hop = q_word_h + hop_ctx
            else:
                q_word_h_hop = q_word_h + hop_ctx + z * q_word_h_dist_ctx[
                    -1]  # [bsz, max_q, max_q]*[bsz, max_q, dim_h] = [bsz, max_q, dim_h]
            q_word_h_dist_ctx.append(hop_ctx + z*q_word_h_dist_ctx[-1])

            q_word_att = torch.sum(q_dist, dim=1, keepdim=True)  # [bsz, 1, max_q]
            q_word_att = torch.softmax(q_word_att, 2)
            q_word_att = q_word_att / (torch.sum(q_word_att, dim=2, keepdim=True) + 1e-6)  # [bsz, max_q, max_q]
            word_attns.append(q_word_att)  # bsz,1,q_max

            ctx_h = (q_word_h_hop.transpose(-1, -2) @ q_word_att.transpose(-1, -2)).squeeze(
                2)  # [bsz, dim_h, max_q] * [bsz, max_q,1]

            ctx_h_list.append(ctx_h) 
            rel_logit = self.rel_classifier(ctx_h)  # [bsz, num_relations]
            rel_dist = torch.sigmoid(rel_logit)
            rel_probs.append(rel_dist)

            last_e = self.follow(last_e, rel_dist)

            # reshape >1 scores to 1 in a differentiable way
            m = last_e.gt(1).float()
            z = (m * last_e + (1 - m)).detach()
            last_e = last_e / z

            # Specifically for MetaQA: reshape cycle entities to 0, because A-r->B-r_inv->A is not allowed
            if t > 0:
                prev_rel = torch.argmax(rel_probs[-2], dim=1)
                curr_rel = torch.argmax(rel_probs[-1], dim=1)
                prev_prev_ent_prob = ent_probs[-2] if len(ent_probs) >= 2 else e_s
                # in our vocabulary, indices of inverse relations are adjacent. e.g., director:0, director_inv:1
                m = torch.zeros((bsz, 1)).to(device)
                m[(torch.abs(prev_rel - curr_rel) == 1) & (torch.remainder(torch.min(prev_rel, curr_rel), 2) == 0)] = 1
                ent_m = m.float() * prev_prev_ent_prob.gt(0.9).float()
                last_e = (1 - ent_m) * last_e

            ent_probs.append(last_e)

        hop_res = torch.stack(ent_probs, dim=1)  # [bsz, num_hop, num_ent]

        ctx_h_history = torch.stack(ctx_h_list, dim=2)  # [bsz, dim_h, num_hop]

        hop_logit = self.hop_att_layer(ctx_h_history.transpose(-1, -2))  # bsz, num_hop, 1
        hop_attn = torch.softmax(hop_logit.transpose(-1, -2), 2).transpose(-1, -2)  # bsz, num_hop, 1

        if not self.training:

            hop_att_tmp = hop_attn.squeeze()
            loc = torch.argmax(hop_att_tmp, dim=-1)
            mask_1hop = 1 - hop_res[:, 0].gt(0.0).float() * loc.gt(1.0).float().unsqueeze(-1)
            discount_1hop = mask_1hop * 0.1 + 0.9 
            last_e = torch.sum(hop_res * hop_attn, dim=1)  # [bsz, num_ent]
            last_e = last_e * discount_1hop 
            return {
                'e_score': last_e,
                'word_attns': word_attns,
                'rel_probs': rel_probs,
                'ent_probs': ent_probs,
                'hop_attn': hop_attn
            }
        else:
            last_e = torch.sum(hop_res * hop_attn, dim=1)  # [bsz, num_ent]
            # Distance loss
            weight = answers * 9 + 1
            loss_score = torch.mean(weight * torch.pow(last_e - answers, 2))
            loss = {'loss_score': loss_score}

            if self.aux_hop:
                loss_hop = nn.CrossEntropyLoss()(hop_logit.squeeze(), hop - 1)
                loss['loss_hop'] = 0.01 * loss_hop

            return loss