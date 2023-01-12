import torch
from torch import nn
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

class DictDataset(torch.utils.data.Dataset):
	def __init__(self, inp, labels, device='cuda', keys=None):
		if keys is None: keys = set(inp.keys())
		self.x = {k:v.to(device) for k,v in inp.items() if k in keys}
		self.labels = labels.to(device)
	def __getitem__(self, idx):
		item = {key:val[idx] for key, val in self.x.items()}
		item['labels'] = self.labels[idx]
		return item
	def __len__(self):
		return len(self.labels)

from transformers import BertTokenizer, BertModel
def GetTokenizer(plm='hfl/chinese-roberta-wwm-ext'):
    return BertTokenizer.from_pretrained(plm)

class BERTClassification(nn.Module):
	def __init__(self, n_tags, cls_only=False, plm='hfl/chinese-roberta-wwm-ext') -> None:
		super().__init__()
		self.n_tags = n_tags
		self.bert = BertModel.from_pretrained(plm)
		self.fc = nn.Linear(768, n_tags)
		self.cls_only = cls_only
	def forward(self, x, seg=None):
		if seg is None: seg = torch.zeros_like(x)
		z = self.bert(x, token_type_ids=seg).last_hidden_state
		if self.cls_only: z = z[:,0]
		out = self.fc(z)
		return out

def pad_to_fixed_length(x, length, value=0):
	s = x.shape
	lpad = length - x.shape[1]
	if lpad > 0: 
		pad = torch.zeros((s[0], lpad)+s[2:], dtype=x.dtype) + value
		x = torch.cat([x, pad], dim=1)
	return x[:,:length]

def train_pt_model(model, train_dl, criterion, optimizer, epochs=3, test_func=None, scheduler=None, data_func=None):
	if data_func is None:
		def data_func1(ditem):
			if type(ditem) is type({}):
				return {k:v.cuda() for k, v in ditem.items() if k != 'labels'}, ditem['labels'].cuda()
			if type(ditem) is type(tuple()) and len(ditem) > 2:
				return [x.cuda() for x in ditem[:-1]], ditem[-1].cuda() 
			return ditem
		data_func = data_func1
	for epoch in range(epochs):
		model.train()
		print(f'\nEpoch {epoch+1} / {epochs}:')
		pbar = tqdm(train_dl)
		iters, accloss = 0, 0
		for ditem in pbar:
			item, label = data_func(ditem)
			item, label = item.cuda(), label.cuda()
			optimizer.zero_grad()
			out = model(item)
			loss = criterion(out, label)
			iters += 1; accloss += loss
			loss.backward()
			optimizer.step()
			if scheduler: scheduler.step()
			pbar.set_postfix({'loss': f'{accloss/iters:.6f}'})
		pbar.close()
		if test_func:
			model.eval()
			test_func()

def train_model(model, optimizer, train_dl, epochs=3, train_func=None, test_func=None, 
				scheduler=None, save_file=None, accelerator=None, epoch_len=None, clean_func=None):
	for epoch in range(epochs):
		model.train()
		if accelerator:
			if accelerator.is_local_main_process: print(f'\nEpoch {epoch+1} / {epochs}:')
			pbar = tqdm(train_dl, total=epoch_len, disable=not accelerator.is_local_main_process)
		else: 
			pbar = tqdm(train_dl, total=epoch_len)
			print(f'\nEpoch {epoch+1} / {epochs}:')
		metricsums = {}
		iters, accloss = 0, 0
		for ditem in pbar:
			metrics = {}
			loss = train_func(model, ditem)
			if type(loss) is type({}):
				metrics = {k:v.detach().mean().item() for k,v in loss.items() if k != 'loss'}
				loss = loss['loss']
			iters += 1; accloss += loss.detach().item()
			optimizer.zero_grad()
			if accelerator: 
				accelerator.backward(loss)
			else: 
				loss.backward()
			optimizer.step()
			if scheduler:
				if accelerator is None or not accelerator.optimizer_step_was_skipped:
					scheduler.step()
			del loss
			if clean_func: clean_func(ditem)
			for k, v in metrics.items(): metricsums[k] = metricsums.get(k,0) + v
			infos = {'loss': f'{accloss/iters:.4f}'}
			for k, v in metricsums.items(): infos[k] = f'{v/iters:.4f}' 
			pbar.set_postfix(infos)
			if epoch_len and iters > epoch_len: break
		pbar.close()
		if save_file:
			if accelerator:
				accelerator.wait_for_everyone()
				unwrapped_model = accelerator.unwrap_model(model)
				accelerator.save(unwrapped_model.state_dict(), save_file)
			else:
				torch.save(model.state_dict(), save_file)
		if test_func:
			if accelerator is None or accelerator.is_local_main_process or True: 
				model.eval()
				test_func()

class MultiBinaryClassification():
	def __init__(self):
		self.cri = nn.BCELoss()
	def get_optim_and_sche(self, model, lr, epochs, dl_train):
		total_steps = epochs * len(dl_train)
		return get_bert_optim_and_sche(model, lr, total_steps)
	def collate_fn(self, items):
		xx = nn.utils.rnn.pad_sequence([x for x,y in items], batch_first=True)
		yy = nn.utils.rnn.pad_sequence([y for x,y in items], batch_first=True)
		return xx, yy.float()
	def train_func(self, model, ditem):
		x, y = ditem[0].cuda(), ditem[1].cuda()
		out = model(x)
		loss = self.cri(out, y)
		oc = (out > 0.5).float()
		prec = (oc + y > 1.5).sum() / max(oc.sum().item(), 1)
		reca = (oc + y > 1.5).sum() / max(y.sum().item(), 1)
		f1 = 2 * prec * reca / (prec + reca)
		return {'loss': loss, 'prec': prec, 'reca': reca, 'f1':f1}
	def dev_func(self, model, dl_dev, return_str=True):
		outs = [];  ys = []
		for x, y in dl_dev:
			out = (model(x.cuda()) > 0.5).long().detach().cpu()
			outs.append(out)
			ys.append(y)
		outs = torch.cat(outs, 0)
		ys = torch.cat(ys, 0)
		accu = (outs == ys).float().mean()
		prec = (outs + ys == 2).float().sum() / outs.sum()
		reca = (outs + ys == 2).float().sum() / ys.sum()
		f1 = 2 * prec * reca / (prec + reca)
		if return_str: return f'Accu: {accu:.4f},  Prec: {prec:.4f},  Reca: {reca:.4f},  F1: {f1:.4f}'
		return accu, prec, reca, f1


def lock_transformer_layers(bert, num_locks):
	import ljqpy
	num = 0
	for name, param in bert.named_parameters():
		if 'embeddings.' in name: ll = -1
		else: 
			ll = int('0'+ljqpy.RM('encoder.layer.([0-9]+)\\.', name))
		if ll < num_locks:
			#print(f'locking {name}')
			num += 1
			param.requires_grad = False
	print(f'Locked {num} parameters ...')

def get_bert_adamw(model, lr=1e-4):
	no_decay = ['bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.01},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
	]
	return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

def get_bert_optim_and_sche(model, lr, total_steps, warmup_steps=-1):
	optimizer = get_bert_adamw(model, lr=lr)
	if warmup_steps < 0: warmup_steps = total_steps//10
	scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
	return optimizer, scheduler

class AvgMetric:
	def __init__(self, n=100):
		self.h = [0]
		self.n = n
	def add(self, x):
		self.h.append(self.h[-1]+x)
		if len(self.h) > self.n*10: self.h = self.h[-self.n*2:]
	def read(self):
		if len(self.h) < (self.n+1): return (self.h[-1] - self.h[-2])
		return (self.h[-1] - self.h[-1-self.n]) / self.n

def cycle(dl):
	while True:
		for x in dl: yield x