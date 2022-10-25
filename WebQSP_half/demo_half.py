import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
# sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../'))) 
# path_abs = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
path_abs = os.path.abspath(os.path.join(os.getcwd(), '../'))
print(path_abs)
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import shutil
from tqdm import tqdm
import numpy as np
import time
from utils.misc import MetricLogger, batch_device, RAdam
from utils.lr_scheduler import get_linear_schedule_with_warmup
from WebQSP_half.data_half_hop import load_data
from WebQSP_half.model_wsp_half import GCF
import logging
from collections import defaultdict
from IPython import embed

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

torch.set_num_threads(1)  # avoid using multiple cpus

import setproctitle

setproctitle.setproctitle("GCF_half_demo") 

def test(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path_abs = '/YourPath/GCF'
    input_dir = path_abs + '/' + args.input_dir
    args.ckpt = path_abs + '/' + args.ckpt
    print(input_dir)
    ent2id, rel2id, triples, train_loader, val_loader = load_data(input_dir, args.bert_name, args.batch_size)
    logging.info("Create model.........")
    model = GCF(args, ent2id, rel2id, triples)
    if not args.ckpt == None:
        model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    model.Msubj = model.Msubj.to(device)
    model.Mobj = model.Mobj.to(device)
    model.Mrel = model.Mrel.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    bert_param = [(n, p) for n, p in model.named_parameters() if n.startswith('bert_encoder')]
    other_param = [(n, p) for n, p in model.named_parameters() if not n.startswith('bert_encoder')]
    print('number of bert param: {}'.format(len(bert_param)))
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.bert_lr},
        {'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.bert_lr},
        {'params': [p for n, p in other_param if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': [p for n, p in other_param if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.lr},
    ]
    if args.opt == 'adam':
        optimizer = optim.Adam(optimizer_grouped_parameters)
    elif args.opt == 'radam':
        optimizer = RAdam(optimizer_grouped_parameters)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(optimizer_grouped_parameters)
    else:
        raise NotImplementedError
    logging.info("Start testing........")

    acc, acc_hop, acc_hop_att, f1 = validate(args, model, val_loader, device)
    logging.info(acc_hop)
    logging.info(acc)
    logging.info(acc_hop_att)
    logging.info(f1)

def validate(args, model, data, device, verbose=False, thresholds=0.985):
    model.eval()
    count = 0
    correct = 0
    hop_count = defaultdict(list)
    hop_att_count = defaultdict(int)
    hop_pred_count = defaultdict(int)
    num_answers_total = 0  # TP + FN
    num_answers_pred_total = 0  # TP + FP
    TP_total = 0
    f = open('wrong.txt', 'w+')
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            outputs = model(*batch_device(batch[:-1], device)) # [bsz, Esize]
            topic_entities, questions, answers, entity_range, hops = batch
            labels = torch.nonzero(answers)
            answer_list = [[] for _ in range(topic_entities.shape[0])]
            for x in labels:
                answer_list[x[0].item()].append(x[1].item())
            num_answers = sum(len(x) for x in answer_list) 
            num_answers_total += num_answers
            e_score = outputs['e_score'].cpu()
            e_score_answers = torch.where(e_score >= thresholds)
            num_pred = e_score_answers[0].shape[0]
            num_answers_pred_total += num_pred

            TP = 0
            for i in range(e_score_answers[0].shape[0]):
                if e_score_answers[1][i].item() in answer_list[e_score_answers[0][i].item()]:
                    TP += 1
            TP_total += TP
            topic_entities_idx = torch.nonzero(topic_entities)
            for item in topic_entities_idx:
                e_score[item[0], item[1]] = 0

            scores, idx = torch.max(e_score, dim=1) # [bsz], [bsz]
            match_score = torch.gather(batch[2], 1, idx.unsqueeze(-1)).squeeze().tolist()
            count += len(match_score)
            correct += sum(match_score)
            for i in range(len(match_score)):
                h_pred = outputs['hop_attn'][i].argmax().item() 
                h = hops[i] - 1 
                hop_count[h].append(match_score[i])
                hop_att_count[h] += (h == h_pred)
                hop_pred_count[h_pred] += 1
            # wrong_list = []
            if not verbose:
                continue
            for i in range(len(match_score)):
                if match_score[i] == 1: # or match_score[i] == 1:
                    print(match_score[i])
                    print('================================================================')
                    question_ids = batch[1]['input_ids'][i].tolist()
                    question_tokens = data.tokenizer.convert_ids_to_tokens(question_ids)
                    question = []
                    for item in question_tokens[1:]:
                        if item =='[SEP]':
                            break
                        question.append(item)
                    question = ' '.join(question)
                    print(question)
                    topic_id = batch[0][i].argmax(0).item()
                    print('> topic entity: {}'.format(data.id2ent[topic_id]))
                    print("> hops: {}".format(hops[i]))
                    for t in range(2):
                        print('>>>>>>> step {}'.format(t))
                        tmp = ' '.join(['{}: {:.2f}'.format(x, y) for x,y in
                            zip(question_tokens, outputs['word_attns'][t][i][0].tolist())])  # [0]
                        print('> Attention: ' + tmp)
                        print('> Relation:')
                        rel_idx = outputs['rel_probs'][t][i].gt(0.00).nonzero().squeeze(1).tolist()
                        for x in rel_idx:
                            print('  {}: {:.3f}'.format(data.id2rel[x], outputs['rel_probs'][t][i][x].item()))
                        for _ in outputs['ent_probs'][t][i].gt(0.0).nonzero().squeeze(1).tolist():
                            print(data.id2ent[_])
                            print(outputs['ent_probs'][t][i][_])
                        print('> Entity: {}'.format('; '.join([data.id2ent[_] for _ in outputs['ent_probs'][t][i].gt(0.8).nonzero().squeeze(1).tolist()])))
                    print('----')
                    print('> max is {}'.format(data.id2ent[idx[i].item()]))
                    print('> golden: {}'.format('; '.join([data.id2ent[_] for _ in answers[i].gt(0.9).nonzero().squeeze(1).tolist()])))
                    print('> prediction: {}'.format('; '.join([data.id2ent[_] for _ in e_score[i].gt(0.9).nonzero().squeeze(1).tolist()])))
                    print(outputs['hop_attn'][i].tolist())
                    h = hops[i]
                    wrong_list = [question, '\t', str(h)]
                    f.write(''.join(wrong_list))
                    f.write('\n')
    f.close()
    acc = correct / count
    acc_hop = ('real hop accuracy: 1-hop {} (total {}), 2-hop {} (total {})'.format(
        sum(hop_count[0]) / (len(hop_count[0]) + 0.1),
        len(hop_count[0]),
        sum(hop_count[1]) / (len(hop_count[1]) + 0.1),
        len(hop_count[1]),
    ))
    acc_hop_att = ('real hop-att accuracy: 1-hop {} (total {}), 2-hop {} (total {})'.format(
        hop_att_count[0] / (len(hop_count[0]) + 0.1),
        hop_pred_count[0],
        hop_att_count[1] / (len(hop_count[1]) + 0.1),
        hop_pred_count[1]
    ))

    precision = TP_total / num_answers_pred_total
    recall = TP_total / num_answers_total
    f1 = 2*precision * recall / (precision + recall + 1e-6)

    f1_info = ("precision: {}, recall: {}, f1: {}".format(precision, recall, f1))

    return acc, acc_hop, acc_hop_att, f1_info

# python demo.py --input_dir data/WebQSP_half --save_dir checkpoints/WebQSP_test --ckpt model_wqsp_half

def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True, help='path to the data')
    parser.add_argument('--save_dir', required=True, default=None, help='path to save checkpoints and logs')
    parser.add_argument('--ckpt', default='checkpoints/WebQSP_half/model_wqsp_half.pt')
    # training parameters
    parser.add_argument('--bert_lr', default=3e-5, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=60, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--opt', default='radam', type=str)
    parser.add_argument('--warmup_proportion', default=0.05, type=float)
    # model parameters
    parser.add_argument('--bert_name', default='bert-base-uncased', choices=['roberta-base', 'bert-base-uncased'])
    args = parser.parse_args()

    # make logging.info display into both shell and file
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    args.log_name = time_ + '_{}_{}_{}.log'.format(args.opt, args.lr, args.batch_size)
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, args.log_name))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k + ':' + str(v))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    test(args)


if __name__ == '__main__':
    main()
