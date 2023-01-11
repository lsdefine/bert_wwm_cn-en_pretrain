import os, sys, ljqpy, random
import zipfile
import numpy as np
import jieba_fast as jieba

datadir = '/mnt/data122/datasets/CLUECorpus2020(100G)'
wikifn = '/mnt/data122/datasets/WikiCorpus/wiki.txt'
pilefn = '/data1/datas/pile/test'

if not os.path.exists(datadir):
    datadir = '/data1/datas/clue/train/'
if not os.path.exists(wikifn):
    wikifn = '/data1/ljq/wikicorpus/wiki.txt'

def LoadListg(fn):
	with open(fn, encoding="utf-8") as fin:
		for ll in fin:
			yield ll.strip()

def MixedSentences():
    zips = ljqpy.ListDirFiles(datadir, lambda x:'train_clue' in x and x.endswith('.zip'))
    if os.path.exists(wikifn): zips.append('wiki')
    if os.path.exists(pilefn): zips.extend(ljqpy.ListDirFiles(pilefn, lambda x:x.endswith('.jsonl'))[:2])
    while True:
        random.shuffle(zips)
        for zipfn in zips:
            if zipfn == 'wiki':
                for x in LoadListg(wikifn): yield x
                yield ''
            elif 'pile' in zipfn:
                for x in ljqpy.LoadJsonsg(zipfn):
                    lines = [z for z in x['text'].strip().split('\n') if z != '']
                    for sent in lines: yield sent
                    yield ''
            else:
                with zipfile.ZipFile(zipfn, 'r') as fzip:
                    ilist = fzip.infolist()
                    random.shuffle(ilist)
                    for zinfo in ilist:
                        with fzip.open(zinfo.filename, 'r') as fin:
                            for x in fin: yield x.decode().strip()
                        yield ''

import torch

class PureGenDataset(torch.utils.data.Dataset):
    def __init__(self, gen, num=100000) -> None:
        self.gen = gen
        self.num = num
    def __len__(self): return self.num
    def __getitem__(self, k):
        return next(self.gen)

def wwm_encode(text, tokenizer, mask=103):
    words = jieba.lcut(text)
    rands = np.random.random(len(words))
    source, target = [], []
    for r, w in zip(rands, words):
        ids = tokenizer.encode(w, add_special_tokens=False)
        if r < 0.15 * 0.8:
            source.extend([mask] * len(ids))
            target.extend(ids)
        elif r < 0.15 * 0.9:
            source.extend(ids)
            target.extend(ids)
        elif r < 0.15:
            source.extend(np.random.choice(tokenizer.vocab_size-1, size=len(ids)) + 1)
            target.extend(ids)
        else:
            source.extend(ids)
            target.extend([-100] * len(ids))
    return source, target

def PadLists(xs, ps, maxlen):
    lacklen = maxlen - len(xs[0])
    if lacklen > 0:
        for i in range(len(xs)):
            xs[i].extend([ps[i]] * lacklen)
    return xs

def CutLists(xs, ps, maxlen):
    for i in range(len(xs)):
        xs[i] = xs[i][:maxlen-1] + [ps[i]]
    return xs
    

def RoBERTaFullSent(gen, tokenizer, maxlen=512, cls=101, sep=102, mask=103, ypad=-100):
    xx, yy = [cls], [ypad]; 
    segs, seg = [0], 0
    while True:
        text = next(gen)
        source, target = wwm_encode(text, tokenizer, mask)
        if len(xx) + len(source) + 1 > maxlen and len(xx) > 1:
            xx, yy, segs = CutLists([xx, yy, segs], [sep, ypad, 0], maxlen)
            xx, yy, segs = PadLists([xx, yy, segs], [0, ypad, 0], maxlen)
            yield {'input_ids':xx, 'labels':yy, 'token_type_ids':segs}
            xx, yy = [cls], [ypad]
            segs, seg = [0], 0
        if seg == 0 and len(xx) > 1 and random.random() < 0.3: 
            xx.append(sep); yy.append(ypad); segs.append(seg)
            seg = 1
        xx.extend(source); yy.extend(target)
        segs.extend([seg] * len(source))

def wwm_info(text, tokenizer):
    words = jieba.lcut(text)
    source, wwm = [], []
    for w in words:
        if len(w) == 1: ids = [tokenizer.convert_single_token(w)]
        else: ids = tokenizer.encode(w, add_special_tokens=False)
        source.extend(ids)
        wwm.extend([1]+[2]*(len(ids)-1))
    return source, wwm

def wwm_mask(xx, wwms, tokenizer, mask=103, prob=0.15):
    newx, newy = [], []
    words = []
    for i, w in enumerate(wwms):
        if w == 2: words[-1] = i+1
        else: words.extend([i, i+1])
    words = [(words[i], words[i+1]) for i in range(0, len(words), 2)]
    rands = np.random.random(len(words))
    for r, (u, v) in zip(rands, words):
        ids = xx[u:v]
        if wwms[u] == 0: r = 100 # dont mask special tokens
        if r < prob * 0.8:
            newx.extend([mask] * len(ids))
            newy.extend(ids)
        elif r < prob * 0.9:
            newx.extend(ids)
            newy.extend(ids)
        elif r < prob:
            newx.extend(np.random.choice(tokenizer.vocab_size-1, size=len(ids)) + 1)
            newy.extend(ids)
        else:
            newx.extend(ids)
            newy.extend([-100] * len(ids))
    return newx, newy

def RoBERTaFullSentFast(gen, tokenizer, maxlen=512, repeat=3, cls=101, sep=102):
    xx, segs, seg, wwms = [cls], [0], 0, [0]
    newpara = False
    while True:
        text = next(gen)
        if text == '':
            newpara = True
            continue
        source, wwm = wwm_info(text, tokenizer)
        if len(xx) + len(source) + 1 > maxlen and len(xx) > 1:
            xx, segs, wwms = CutLists([xx, segs, wwms], [sep, seg, 0], maxlen)
            xx, segs, wwms = PadLists([xx, segs, wwms], [0, 0, 0], maxlen)
            for _ in range(repeat):
                newx, newy = wwm_mask(xx, wwms, tokenizer)
                yield {'input_ids':newx, 'labels':newy, 'token_type_ids':segs}
            xx, segs, seg, wwms = [cls], [0], 0, [0]
        if newpara:
            xx.append(sep); segs.append(seg); wwms.append(0); seg = 1
            newpara = False
        xx.extend(source); segs.extend([seg] * len(source))
        wwms.extend(wwm)

def NSPGenerator(gen, tokenizer, maxlen=512, cls=101, sep=102, repeat=0):
    pools = []
    for _ in range(3):
        z = next(gen)
        if z != '': pools.append(z)
    continuous = False
    while True:
        text = next(gen)
        if text == '': 
            continuous = False
            continue
        pools.append(text)
        if len(pools) > 100000: pools = pools[10000:20000] + pools[-10000:]
        nsp_label = random.randint(0, 1)
        if not continuous: nsp_label = 0
        textb = text
        if nsp_label == 1: texta = pools[-2]
        else: texta = random.choice(pools[:-2])
        xx, segs, wwms = [cls], [0], [0]
        source, wwm = wwm_info(texta, tokenizer)
        xx.extend(source); segs.extend([0] * len(source)); wwms.extend(wwm)
        xx.append(sep); segs.append(0); wwms.append(0)
        source, wwm = wwm_info(textb, tokenizer)
        xx.extend(source); segs.extend([1] * len(source)); wwms.extend(wwm)
        xx, segs, wwms = CutLists([xx, segs, wwms], [sep, 1, 0], maxlen)
        xx, segs, wwms = PadLists([xx, segs, wwms], [0, 0, 0], maxlen)
        newx, newy = wwm_mask(xx, wwms, tokenizer)
        yield {'input_ids':newx, 'labels':newy, 'token_type_ids':segs, 'nsp_label':nsp_label}
        continuous = True      

if __name__ == '__main__':
    from cetokenizer import CEBertTokenizerFast
    #tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    tokenizer = CEBertTokenizerFast('vocab.txt')
    #rr = wwm_encode('努力！未来 lucky star！normalization normalization normalization', tokenizer)
    #print(rr)
    #print(tokenizer.convert_ids_to_string(rr[0]))
    #sg = MixedSentences()
    #for i in range(10):
    #    s = next(sg)
    #    print(s[:43])
    #sys.exit()
    ds = PureGenDataset(NSPGenerator(MixedSentences(), tokenizer, maxlen=256, repeat=1), 3)
    #print(len(ds))
    for i in range(3): 
        z = ds[i]
        print('')
        print(z['nsp_label'])
        print(tokenizer.convert_ids_to_string(z['input_ids'], no_pad=True))
    print('done')