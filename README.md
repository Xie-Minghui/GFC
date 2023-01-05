# GFC
Pytorch implementation for EMNLP 2022 paper

**[A Sequential Flow Control Framework for Multi-hop Knowledge Base Question Answering](https://preview.aclanthology.org/emnlp-22-ingestion/2022.emnlp-main.578.pdf)**

Minghui Xie, Chuzhan Hao, and Peng Zhang

Overall architecture of our proposed GFC
model.
<div align="center">
    <img src="/pics/framework.png" width="30%">
</div><br/>

The schematic diagram of the GRU-inspired
Flow Control Framework.

<div align="center">
    <img src="/pics/model.png" width="70%">
</div><br/>

If you find this code useful in your research, please cite
```bib
@InProceedings{xie2022gfc,
  author =  {Minghui Xie and Chuzhan Hao and Peng Zhang},
  title =   {A Sequential Flow Control Framework for Multi-hop Knowledge Base Question Answering},
  year =    {2022},  
  booktitle = {Empirical Methods in Natural Language Processing (EMNLP)},  
} 
```

## dependencies
- pytorch>=1.2.0
- [transformers](https://github.com/huggingface/transformers)
- tqdm
- nltk
- shutil

## Prepare Datasets
For all raw data files and their corresponding preprocessed data files,
we have uploaded them to [google drive](https://drive.google.com/drive/folders/1ur-tSF_A1AkQWLkARDSMMQYBGODjKzwY?usp=sharing)

### [WebQSP](https://drive.google.com/drive/folders/1RlqGBMo45lTmWz9MUPTq-0KcjSd3ujxc?usp=sharing)
We use script files from [EmbedKGQA](https://github.com/malllabiisc/EmbedKGQA)
to process completed dataset of WebQSP.

### WebQSP-half
We use the dataset preprocessed by [EmbedKGQA](https://github.com/malllabiisc/EmbedKGQA).

### MetaQA
You should preprocess MetaQA using MetaQA/preprocess.py


### [CompWebQ](https://drive.google.com/file/d/1ua7h88kJ6dECih6uumLeOIV9a3QNdP-g/view?usp=sharing)
We use dataset processed by [NSM](https://github.com/RichardHGL/WSDM2021_NSM).

## Demo
To ensure the reproducibility, we write demo.py for all 4 main tasks.
You can use pretrained model checkpoints to reproduce the results. You should put the checkpoints files into
corresponding path.
The checkpoints files are uploaded to [google drive](https://drive.google.com/drive/folders/1s0SwIALbgpJfaT800TEKwp8P2YOMc09A?usp=sharing).


### WebQSP
Enter the directory GFC/WebQSP, then input the following command.

```shell
python demo_wsp.py --input_dir data/WebQSP --save_dir checkpoints/WebQSP --ckpt checkpoints/WebQSP/model_wqsp.pt
```
### WebQSP_half
Enter the directory GFC/WebQSP_half, then input the following command
```shell
python demo_half.py --input_dir data/WebQSP_half --save_dir checkpoints/WebQSP_half --ckpt checkpoints/WebQSP_half/model_wqsp_half.pt
```

### CompWebQ
Enter the directory GFC/CWQ, then input the following command
```shell
python demo_cwq.py --input_dir data/CWQ --save_dir checkpoints/CWQ --ckpt checkpoints/CWQ/model_cwq.pt
```

### MetaQA
Enter the directory GFC/CWQ, then input the following command
```shell
python demo_metaqa.py --input_dir data/Metaqa --save_dir checkpoints/Metaqa_ps --ckpt checkpoints/Metaqa/model_metaqa.pt
```


## Experiments
We train and test simultaneously.
You should enter the corresponding directory of different datasets.
### WebQSP
Enter the directory GFC/WebQSP, then input the following command.
```shell
python train_hop_final.py --input_dir data/WebQSP --save_dir checkpoints/WebQSP
```

### WebQSP_half
Enter the directory GFC/WebQSP_half, then input the following command
```shell
python train_half_hop_final.py --input_dir data/WebQSP_half --save_dir checkpoints/WebQSP_half
```

### CompWebQ
Enter the directory GFC/CWQ, then input the following command.
```shell
python train_final.py --input_dir data/CWQ --save_dir checkpoints/CWQ --rev
```

### MetaQA
Enter the directory GFC/MetaQA, then input the following command.
```shell
python train_final.py --glove_pt data/glove/glove.840B.300d.pickle --input_dir data/MetaQA --save_dir checkpoints/MetaQA
```
## Acknowledgement
This repo is built upon the following work:
```
TransferNet: An Effective and Transparent Framework for Multi-hop Question Answering over Relation Graph. Jiaxin Shi, Shulin Cao, Lei Hou1∗, Juanzi Li1 and Hanwang Zhang. EMNLP 2021.
https://github.com/shijx12/TransferNet
```
Many thanks to the authors and developers!