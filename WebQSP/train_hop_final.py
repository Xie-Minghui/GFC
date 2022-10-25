import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
from WebQSP.data_hop_final import load_data
from WebQSP.model_wsp import GCF
from WebQSP.predict_f1_hop import validate
from transformers import AdamW
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

torch.set_num_threads(1)  # avoid using multiple cpus

import setproctitle

setproctitle.setproctitle("GCF_WSP")


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path_abs = '/YourPath/GCF'
    input_dir = path_abs + '/' + args.input_dir
    # input_dir = args.input_dir
    print(input_dir)
    ent2id, rel2id, triples, train_loader, val_loader = load_data(input_dir, args.bert_name, args.batch_size)
    logging.info("Create model.........")
    model = GCF(args, ent2id, rel2id, triples)
    if not args.ckpt == None:
        model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    # model.triples = model.triples.to(device)
    model.Msubj = model.Msubj.to(device)
    model.Mobj = model.Mobj.to(device)
    model.Mrel = model.Mrel.to(device)

    t_total = len(train_loader) * args.num_epoch
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
    # optimizer_grouped_parameters = [{'params':model.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr}]
    if args.opt == 'adam':
        optimizer = optim.Adam(optimizer_grouped_parameters)
    elif args.opt == 'radam':
        optimizer = RAdam(optimizer_grouped_parameters)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(optimizer_grouped_parameters)
    else:
        raise NotImplementedError
    args.warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    meters = MetricLogger(delimiter="  ")
    logging.info("Start training........")
    print(args.num_epoch)
    for epoch in range(args.num_epoch):
        model.train()
        for iteration, batch in enumerate(train_loader):
            iteration = iteration + 1
            loss = model(*batch_device(batch[:-1], device))
            optimizer.zero_grad()
            if isinstance(loss, dict):
                if len(loss) > 1:
                    total_loss = sum(loss.values())
                else:
                    total_loss = loss[list(loss.keys())[0]]
                meters.update(**{k: v.item() for k, v in loss.items()})
            else:
                total_loss = loss
                meters.update(loss=loss.item())
            total_loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.5)
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            scheduler.step()

            if iteration % (len(train_loader) // 10) == 0:
                # if True:

                logging.info(
                    meters.delimiter.join(
                        [
                            "progress: {progress:.3f}",
                            "{meters}",
                            "lr: {lr:.6f}",
                        ]
                    ).format(
                        progress=epoch + iteration / len(train_loader),
                        meters=str(meters),
                        lr=optimizer.param_groups[2]["lr"],
                    )
                )
        if (epoch + 1) % 1 == 0:
            acc, acc_hop, acc_hop_att, f1 = validate(args, model, val_loader, device)
            logging.info(acc_hop)
            logging.info(acc)
            logging.info(acc_hop_att)
            logging.info(f1)
            # torch.save(model.state_dict(), os.path.join(args.save_dir, 'model-{}-{:.4f}.pt'.format(epoch, acc)))

# python train_hop_final.py --input_dir data/WebQSP --save_dir checkpoints/WebQSP
def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True, help='path to the data')
    parser.add_argument('--save_dir', required=True, help='path to save checkpoints and logs')
    parser.add_argument('--ckpt', default=None)
    # training parameters
    parser.add_argument('--bert_lr', default=3e-5, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=60, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--seed', type=int, default=444, help='random seed')
    parser.add_argument('--opt', default='radam', type=str)
    parser.add_argument('--warmup_proportion', default=0.05, type=float)
    # model parameters
    parser.add_argument('--bert_name', default='bert-base-uncased', choices=['roberta-base', 'bert-base-uncased'])
    parser.add_argument('--aux_hop', type=int, default=1, choices=[0, 1],
                        help='utilize question hop to constrain the probability of self relation')
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

    train(args)


if __name__ == '__main__':
    main()
