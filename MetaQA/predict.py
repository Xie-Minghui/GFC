import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict
from utils.misc import MetricLogger, load_glove, idx_to_one_hot
from .data import DataLoader
from .model_metaqa import GCF

from IPython import embed


def validate(args, model, data, device, verbose = False):
    vocab = data.vocab
    model.eval()
    hop_count = defaultdict(int)
    count = defaultdict(int)
    correct = defaultdict(int)
    hop_att_count = defaultdict(int)
    hop_pred_count = defaultdict(int)
    hop_loss_total = 0

    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            questions, topic_entities, answers, hops = batch
            topic_entities = idx_to_one_hot(topic_entities, len(vocab['entity2id']))
            answers = idx_to_one_hot(answers, len(vocab['entity2id']))
            answers[:, 0] = 0
            questions = questions.to(device)
            topic_entities = topic_entities.to(device)
            hops = hops.tolist()
            outputs = model(questions, topic_entities) # [bsz, Esize]
            e_score = outputs['e_score'].cpu()
            topic_entities_idx = torch.nonzero(topic_entities)
            for item in topic_entities_idx:
                e_score[item[0], item[1]] = 0


            scores, idx = torch.max(e_score, dim = 1) # [bsz], [bsz]
            match_score = torch.gather(answers, 1, idx.unsqueeze(-1)).squeeze().tolist()

            for h, m, i in zip(hops, match_score, range(len(hops))):
                count['all'] += 1
                count['{}-hop'.format(h)] += 1
                correct['all'] += m
                correct['{}-hop'.format(h)] += m

                h -= 1 
                h_pred = outputs['hop_attn'][i].argmax().item()
                hop_att_count[h] += (h == h_pred)
                hop_pred_count[h_pred] += 1
                hop_count[h] += 1
            if verbose: 
                for i, m in enumerate(match_score):
                    if hops[i] != 3 or m == 1:
                        continue
                    print(m)
                    print('================================================================')
                    question = ' '.join([vocab['id2word'][_] for _ in questions.tolist()[i] if _ > 0])
                    print(question)
                    print('hop: {}'.format(hops[i]))
                    print('> topic entity: {}'.format(vocab['id2entity'][topic_entities[i].max(0)[1].item()]))
                    for t in range(args.num_steps):
                        print('> > > step {}'.format(t))
                        print(outputs['hop_attn'][i])
                        tmp = ' '.join(['{}: {:.3f}'.format(vocab['id2word'][x], y) for x,y in 
                            zip(questions[i].tolist(), outputs['word_attns'][t][i,0].tolist())
                            if x > 0])
                        print('> ' + tmp)
                        tmp = ' '.join(['{}: {:.3f}'.format(vocab['id2relation'][x], y) for x,y in 
                            enumerate(outputs['rel_probs'][t].tolist()[i])])
                        print('> ' + tmp)
                        print('> entity: {}'.format('; '.join([vocab['id2entity'][_] for _ in range(len(answers[i])) if outputs['ent_probs'][t][i][_].item() > 0.9])))
                    print('----')
                    print('> max is {}'.format(vocab['id2entity'][idx[i].item()]))
                    print('> golden: {}'.format('; '.join([vocab['id2entity'][_] for _ in range(len(answers[i])) if answers[i][_].item() == 1])))
                    print('> prediction: {}'.format('; '.join([vocab['id2entity'][_] for _ in range(len(answers[i])) if e_score[i][_].item() > 0.9])))
                    embed()

    acc = {k:correct[k]/count[k] for k in count} 
    result = ' | '.join(['%s:%.4f'%(key, value) for key, value in acc.items()])
    print(result)

    acc_hop_att = ('real hop-att accuracy: 1-hop {} (total {}), 2-hop {} (total {}), 3-hop {} (total {})'.format(
        hop_att_count[0] / (hop_count[0] + 0.1),
        hop_pred_count[0],
        hop_att_count[1] / (hop_count[1] + 0.1),
        hop_pred_count[1],
        hop_att_count[2] / (hop_count[2] + 0.1),
        hop_pred_count[2]
    ))
    # print(acc_hop_att)
    # print("hop loss: {}".format(hop_loss_total/ count['all']))
    return acc, acc_hop_att


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', default = './input')
    parser.add_argument('--ckpt', required = True)
    parser.add_argument('--mode', default='val', choices=['val', 'vis', 'test'])
    # model hyperparameters
    parser.add_argument('--num_steps', default=3, type=int)
    parser.add_argument('--dim_word', default=300, type=int)
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--aux_hop', type=int, default=1, choices=[0, 1], help='utilize question hop to constrain the probability of self relation')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    val_pt = os.path.join(args.input_dir, 'val.pt')
    test_pt = os.path.join(args.input_dir, 'test.pt')
    val_loader = DataLoader(vocab_json, val_pt, 64, True)
    test_loader = DataLoader(vocab_json, test_pt, 64)
    vocab = val_loader.vocab

    model = GTA(args, args.dim_word, args.dim_hidden, vocab)
    missing, unexpected = model.load_state_dict(torch.load(args.ckpt), strict=False)
    if missing:
        print("Missing keys: {}".format("; ".join(missing)))
    if unexpected:
        print("Unexpected keys: {}".format("; ".join(unexpected)))
    model = model.to(device)
    model.kg.Msubj = model.kg.Msubj.to(device)
    model.kg.Mobj = model.kg.Mobj.to(device)
    model.kg.Mrel = model.kg.Mrel.to(device)

    num_params = sum(np.prod(p.size()) for p in model.parameters())
    print('number of parameters: {}'.format(num_params))

    if args.mode == 'vis':
        validate(args, model, val_loader, device, True)
    elif args.mode == 'val':
        validate(args, model, val_loader, device, False)
    elif args.mode == 'test':
        validate(args, model, test_loader, device, False)

if __name__ == '__main__':
    main()
