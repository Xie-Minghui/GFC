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
import numpy as np
import argparse
import shutil
from tqdm import tqdm
import time
from utils.misc import MetricLogger, load_glove, idx_to_one_hot, RAdam
from MetaQA.data import DataLoader
from MetaQA.model_metaqa import GCF
from MetaQA.predict import validate
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

import setproctitle
setproctitle.setproctitle("GCF_MetaQA")

torch.set_num_threads(1) # avoid using multiple cpus

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.input_dir = path_abs + '/' + args.input_dir
    # args.glove_pt = path_abs + '/' + args.glove_pt
    args.glove_pt = '/YourPath/GCF/data/glove/glove.840B.300d.pickle'
    if 'half' in args.input_dir:
        logging.info('Running on half kb')

    logging.info("Create train_loader, val_loader and test_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    train_pt = os.path.join(args.input_dir, 'train.pt')
    val_pt = os.path.join(args.input_dir, 'val.pt')
    test_pt = os.path.join(args.input_dir, 'test.pt')
    train_loader = DataLoader(vocab_json, train_pt, args.batch_size, args.ratio, training=True)
    val_loader = DataLoader(vocab_json, val_pt, args.batch_size)
    test_loader = DataLoader(vocab_json, test_pt, args.batch_size)
    vocab = train_loader.vocab

    logging.info("Create model.........")
    pretrained = load_glove(args.glove_pt, vocab['id2word'])
    model = GCF(args, args.dim_word, args.dim_hidden, vocab)
    model.word_embeddings.weight.data = torch.Tensor(pretrained) 
    if not args.ckpt == None:
        missing, unexpected = model.load_state_dict(torch.load(args.ckpt), strict=False) 
        if missing:
            logging.info("Missing keys: {}".format("; ".join(missing)))
        if unexpected:
            logging.info("Unexpected keys: {}".format("; ".join(unexpected)))
    model = model.to(device)
    model.kg.Msubj = model.kg.Msubj.to(device)
    model.kg.Mobj = model.kg.Mobj.to(device)
    model.kg.Mrel = model.kg.Mrel.to(device)

    if args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'radam':
        optimizer = RAdam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[3], gamma=0.1)

    meters = MetricLogger(delimiter="  ")
    logging.info("Start training........")

    for epoch in range(args.num_epoch):
        model.train()
        for iteration, batch in enumerate(train_loader):
            
            question, topic_entity, answer, hop = batch
            question = question.to(device)
            topic_entity = idx_to_one_hot(topic_entity, len(vocab['entity2id'])).to(device)
            answer = idx_to_one_hot(answer, len(vocab['entity2id'])).to(device)
            answer[:, 0] = 0 
            hop = hop.to(device)
            loss = model(question, topic_entity, answer, hop)
            optimizer.zero_grad()
            if isinstance(loss, dict):
                total_loss = sum(loss.values())
                meters.update(**{k:v.item() for k,v in loss.items()})
            else:
                total_loss = loss
                meters.update(loss=loss.item())
            total_loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.5)
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()

            if iteration % (len(train_loader) // 100) == 0:
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
                        lr=optimizer.param_groups[0]["lr"],
                    )
                )
        if (epoch + 1) % int(1 / args.ratio) == 0:
            acc, acc_hop_att = validate(args, model, val_loader, device)  # verbose=True
            logging.info(acc)
            logging.info(acc_hop_att)
            scheduler.step()
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_epoch-{}_acc-{:.4f}.pt'.format(epoch, acc['all'])))


def test(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.input_dir = path_abs + '/' + args.input_dir
    # args.glove_pt = path_abs + '/' + args.glove_pt
    args.glove_pt = '/YourPath/GCF/data/glove/glove.840B.300d.pickle'
    if 'half' in args.input_dir:
        logging.info('Running on half kb')

    logging.info("Create train_loader, val_loader and test_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    train_pt = os.path.join(args.input_dir, 'train.pt')
    val_pt = os.path.join(args.input_dir, 'val.pt')
    test_pt = os.path.join(args.input_dir, 'test.pt')
    train_loader = DataLoader(vocab_json, train_pt, args.batch_size, args.ratio, training=True)
    val_loader = DataLoader(vocab_json, val_pt, args.batch_size)
    test_loader = DataLoader(vocab_json, test_pt, args.batch_size)
    vocab = train_loader.vocab

    logging.info("Create model.........")
    pretrained = load_glove(args.glove_pt, vocab['id2word'])
    model = GCF(args, args.dim_word, args.dim_hidden, vocab)
    model.word_embeddings.weight.data = torch.Tensor(pretrained)
    if not args.ckpt == None:
        missing, unexpected = model.load_state_dict(torch.load(args.ckpt), strict=False) 
        if missing:
            logging.info("Missing keys: {}".format("; ".join(missing)))
        if unexpected:
            logging.info("Unexpected keys: {}".format("; ".join(unexpected)))
    model = model.to(device)
    model.kg.Msubj = model.kg.Msubj.to(device)
    model.kg.Mobj = model.kg.Mobj.to(device)
    model.kg.Mrel = model.kg.Mrel.to(device)

    acc, acc_hop_att = validate(args, model, test_loader, device)  # verbose=True
    logging.info(acc)
    logging.info(acc_hop_att)

# python -m MetaQA.train --glove_pt data/glove/glove.840B.300d.pickle --input_dir data/MetaQA_ps --save_dir checkpoints/MetaQA
def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--save_dir', required=True, help='path to save checkpoints and logs')
    parser.add_argument('--glove_pt', required=True)
    parser.add_argument('--ckpt', default = None)
    # training parameters
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=30, type=int) 
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--opt', default='radam', type = str)
    parser.add_argument('--ratio', default=1.0, type=float)
    # model hyperparameters
    parser.add_argument('--num_steps', default=3, type=int)
    parser.add_argument('--dim_word', default=300, type=int)
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--aux_hop', type=int, default=1, choices=[0, 1], help='utilize question hop to constrain the probability of self relation')
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
        logging.info(k+':'+str(v))

    if args.ratio < 1:
        args.num_epoch = int(args.num_epoch / args.ratio)
        logging.info('Due to partial training examples, the actual num_epoch is set to {}'.format(args.num_epoch))

    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)
    test(args)


if __name__ == '__main__':
    main()
