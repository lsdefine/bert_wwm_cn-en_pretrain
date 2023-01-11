import os, sys, time, math
from tqdm import tqdm
import numpy as np
dir = os.path.split(os.path.realpath(__file__))[0]
sys.path.append('/data1/ljq/')
sys.path.append('/home/jqliang/')
import pt_utils

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertForMaskedLM, BertConfig, BertForPreTraining

os.environ['CUDA_VISIBLE_DEVICES'] = '3,5,7'

from cetokenizer import CEBertTokenizer
tokenizer = CEBertTokenizer('vocab.txt')
config = BertConfig.from_pretrained('hfl/chinese-roberta-wwm-ext')
config.vocab_size = len(tokenizer.vocab)
model = BertForPreTraining(config)

maxlen = 256

def collate_fn(xs):
	return {k:torch.tensor([x[k] for x in xs], dtype=torch.long) for k in xs[0].keys()}

if __name__ == '__main__':
	from corpus_dataset import PureGenDataset, NSPGenerator, MixedSentences, RoBERTaFullSentFast
	bsz = 64
	ds = PureGenDataset(NSPGenerator(MixedSentences(), tokenizer, maxlen, repeat=3), bsz*30000)
	dl = torch.utils.data.DataLoader(ds, batch_size=bsz, collate_fn=collate_fn, num_workers=2)
	mfile = 'myroberta_mlm.pt'
	model.bert.load_state_dict(torch.load('myroberta_v1.pt'), strict=False)
	
	#omodel = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
	#sd = {x:y for x,y in omodel.state_dict().items() if 'token_type' in x}
	#model.load_state_dict(sd, strict=False)
	#torch.save(model.bert.state_dict(), 'myroberta.pt')
	#sys.exit()
	epochs = 5
	total_steps = len(dl) * epochs

	import accelerate
	from accelerate import Accelerator, DistributedDataParallelKwargs
	accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)])

	#optimizer, scheduler = pt_utils.get_bert_optim_and_sche(model, 1e-4, total_steps)
	no_decay = ['bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.01},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
	]
	optimizer = accelerate.utils.DummyOptim(optimizer_grouped_parameters)
	scheduler = accelerate.utils.DummyScheduler(optimizer, total_num_steps=total_steps, warmup_num_steps=total_steps//10)
	model, optimizer, dl_train, scheduler = accelerator.prepare(model, optimizer, dl, scheduler)

	device = accelerator.device

	def train_func(model, x):
		attent_mask = (x['input_ids'] > 0).long()
		out = model(input_ids=x['input_ids'].to(device), 
				attention_mask=attent_mask.to(device),
				token_type_ids=x['token_type_ids'].to(device),
				labels=x['labels'].to(device),
				next_sentence_label=x['nsp_label'].to(device),
				return_dict=True)
		return out.loss

	def test_func(): pass

	pt_utils.train_model(model, optimizer, dl, epochs, train_func, test_func, 
				scheduler=scheduler, save_file=mfile, accelerator=accelerator)

	torch.save(model.bert.state_dict(), 'myroberta.pt')
	print('done')