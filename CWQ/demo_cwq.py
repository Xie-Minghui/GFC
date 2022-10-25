import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../'))) 
path_abs = os.path.abspath(os.path.join(os.getcwd(), '../'))
# sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
# path_abs = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print(path_abs)
import time
import torch
import torch.nn as nn
import argparse
import numpy as np
from utils.misc import MetricLogger, batch_device
from CWQ.data import load_data
from CWQ.model_cwq import GCF
from CWQ.predict import validate
from transformers import AdamW  # , get_linear_schedule_with_warmup
from utils.lr_scheduler import get_linear_schedule_with_warmup
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

torch.set_num_threads(1) # avoid using multiple cpus

import setproctitle

setproctitle.setproctitle("GCF_CWQ")

def test(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path_abs = '/YourPath/GCF'
    args.input_dir = path_abs + '/' + args.input_dir
    args.ckpt = path_abs + '/' + args.ckpt
    ent2id, rel2id, train_loader, val_loader, test_loader = load_data(args.input_dir, args.bert_name, args.batch_size, args.rev)
    logging.info("Create model.........")
    model = GCF(args, ent2id, rel2id)
    if not args.ckpt == None:
        model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)

    t_total = len(train_loader) * args.num_epoch
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param = [(n,p) for n,p in model.named_parameters() if n.startswith('bert_encoder')]
    other_param = [(n,p) for n,p in model.named_parameters() if not n.startswith('bert_encoder')]
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

    optimizer = AdamW(optimizer_grouped_parameters)
    args.warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    meters = MetricLogger(delimiter="  ")

    if args.rev:
        logging.info("use reversed relations")
    logging.info("Start training........")

    val_acc = validate(args, model, val_loader, device)
    test_acc = validate(args, model, test_loader, device)
    logging.info('val acc: {:.4f}, test acc: {:.4f}'.format(val_acc, test_acc))
    # torch.save(model.state_dict(), os.path.join(args.save_dir, 'model-{}-{:.4f}.pt'.format(epoch, val_acc)))

# python train_final.py --input_dir data/CWQ --save_dir checkpoints/CWQ --rev
def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True, help='path to the data')
    parser.add_argument('--save_dir', required=True, help='path to save checkpoints and logs')
    parser.add_argument('--ckpt', default='checkpoints/CWQ/model_cwq.pt')
    # training parameters
    parser.add_argument('--bert_lr', default=3e-5, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=60, type=int)  # 30到60
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', type=int, default=444, help='random seed')
    parser.add_argument('--warmup_proportion', default=0.05, type = float)
    # model parameters
    parser.add_argument('--rev', action='store_true', help='whether add reversed relations')
    parser.add_argument('--num_ways', default=1, type=int)
    parser.add_argument('--num_steps', default=2, type=int)
    parser.add_argument('--bert_name', default='bert-base-uncased', choices=['roberta-base', 'bert-base-cased', 'bert-base-uncased'])
    parser.add_argument('--opt', default='radam', type=str)
    args = parser.parse_args()

    # make logging.info display into both shell and file
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    args.log_name = time_ + '_{}_{}_{}.log'.format(args.opt, args.lr, args.batch_size)
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, args.log_name))
    # fileHandler = logging.FileHandler(os.path.join(args.save_dir, 'log.txt'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k+':'+str(v))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    test(args)


if __name__ == '__main__':
    main()
