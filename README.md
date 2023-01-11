# bert_wwm_cn-en_pretrain

Pretrain my own bert/roberta model with my own tokenizer.

Basing on pytorch and HuggingFace Accelerate and Deepspeed.

## CEBertTokenizer
Basing on HuggingFace BertTokenizer, you can follow it for your own tokenizer.
* The same interface.
* The fast version is as fast as HuggingFace BertTokenizerFast.
* All numbers and chinese characters are splitted. 2077->2/0/2/2 vs 207/7 (BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext'))。
* Keep spaces for alignment, but they are replaced with □.
* Add some frequent characters that are [UNK] in hfl/chinese-roberta-wwm-ext, such as "…".
---
* 和HuggingFace BertTokenizer兼容，Fast版本速度和原版相当。
* 所有中文字符和数字进行了切分，2077->2/0/2/2, 原版分词器分为207/7。
* 保留了空格使切分的Token序列和原文本对齐，并将空格全部替换为特殊符号□。（某些下游任务的匹配过程中需要注意空格和□的匹配。）
* 增加了中文文本中常见，但在原分词器中为[UNK]的符号，如省略号…

## Pretraining
WWM (1m steps) then WWM+NSP (500k steps)

Datasets: CLUE100G + Chinese Wiki + part of Pile.

## Comparison in CLUE
![alt](http://kw.fudan.edu.cn/resources/assets/img/ljq_roberta.png)

## Pretrained Model Download
[预训练参数和词表v4下载](http://kw.fudan.edu.cn/apis/fint5/)

## Usage
```
from cetokenizer import CEBertTokenizer, CEBertTokenizerFast
from transformers import BertTokenizer, BertModel, BertConfig
tokenizer = CEBertTokenizerFast('vocab.txt')
config = BertConfig.from_pretrained('hfl/chinese-roberta-wwm-ext')
config.vocab_size = len(tokenizer.vocab)
model = BertModel(config)
model.load_state_dict(torch.load('kw_roberta_ce_v4.pt'), strict=False)
```