import os, sys, time, re
from transformers import BertTokenizer, BertTokenizerFast, PreTrainedTokenizer

def make_merged_vocab():
    import ljqpy
    ct = CEBertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    et = CEBertTokenizer.from_pretrained('bert-base-uncased')
    lst = []
    print(len(ct.vocab))
    print(len(et.vocab))
    evocs = [token for token,i in sorted(et.vocab.items(), key=lambda kv: kv[1])]
    evocs = [t for t in evocs if re.match('^[#a-z]{2,}$', t)]
    evocset = set(evocs)
    #ljqpy.SaveList(evocs, 'ee.txt')
    for token, token_index in sorted(ct.vocab.items(), key=lambda kv: kv[1]):
        if re.match('^[#0-9]{2,}$', token): continue
        if re.search('[0-9]', token) and len(token) > 1 and '[' not in token: continue
        if re.match('^[#\u4e00-\u9fa5]{2,}$', token) and '[' not in token: continue
        if re.match('^([^\x00-\xff]|#){2,}$', token) and len(token.strip('#')) > 1 and '[' not in token: continue
        if token in evocs: continue
        lst.append(token)
    print(len(lst))
    lst += evocs
    print(len(lst))
    puncs = '''@#$%^&()*+-<=>|~《》/\“”【】‘’….。，！？、：；"—'''
    for x in puncs:
        if x not in lst:
            print('not found!', x)
            lst.append(x)
    nnext = (len(lst)//1000+1)*1000 - len(lst)
    for x in range(100, 100+nnext):
        lst.append(f'[unused{x}]')
    print(len(lst))
    ljqpy.SaveList(lst, 'vocab.txt')

def pre_tokenize(sent):
    ss = [];  last = 0
    for i, ch in enumerate(sent):
        if '0' <= ch <= '9' or '\u4e00' <= ch <= '\u9fff': 
            if last < i: ss.append(sent[last:i])
            ss.append(ch)
            last = i+1
    if last < len(sent): ss.append(sent[last:])
    return ss        

def pre_process(text):
    output = []
    for char in text:
        if '0' <= char <= '9':
            output.append(f" {char} ")
        elif char == ' ': output.append(" □ ")
        else: output.append(char)
    return "".join(output)


class CETokenizerBase:
    def restore_token_list(self, text, tokens):
        if tokens[0] == '[CLS]': tokens = tokens[1:-1]
        otokens = []; offset = 0
        for x in tokens:
            if x == '[UNK]': x = text[offset]
            elif x == '□': x = ' '
            elif x.startswith('##'): x = x[2:]
            otokens.append(text[offset:offset+len(x)])
            offset += len(x)
        assert ''.join(otokens) == text
        return otokens  
    def convert_tokens_to_string(self, tokens):
        out_string = "".join(tokens).replace("##", "").replace('□', ' ').strip()
        return out_string
    def convert_ids_to_string(self, ids, no_pad=False):
        if no_pad: ids = [x for x in ids if x > 0]
        return self.convert_tokens_to_string(self.convert_ids_to_tokens(ids))

class CEBertTokenizer(BertTokenizer, CETokenizerBase):
    def _tokenize(self, text):
        return super()._tokenize(pre_process(text))
    def convert_single_token(self, w):
        if w == ' ': return self._convert_token_to_id('□')
        return self._convert_token_to_id(w.lower())
  
class CEBertTokenizerFast(BertTokenizerFast, CETokenizerBase):
    def my_preprocess(self, text_or_pair):
        if isinstance(text_or_pair, (list, tuple)): return tuple(pre_process(x) for x in text_or_pair)
        return pre_process(text_or_pair)
    def _batch_encode_plus(self, batch_text_or_text_pairs, **kwargs):
        batch_input = [self.my_preprocess(x) for x in batch_text_or_text_pairs]
        return super()._batch_encode_plus(batch_input, **kwargs)
    def convert_single_token(self, w):
        if w == ' ': return self._convert_token_to_id_with_added_voc('□')
        return self._convert_token_to_id_with_added_voc(w.lower())
   
def test_tokenize(tokenizer, sents):
    tic = time.time()
    ret = []
    for sent in sents: ret.append(tokenizer.encode(sent))
    for u, v in zip(sents, sents[1:]): ret.append(tokenizer.encode(u, v))
    ret.extend(tokenizer(sents)['input_ids'])
    ret.extend(tokenizer(sents)['token_type_ids'])
    ret.extend(tokenizer(sents, sents[1:]+sents[:1])['input_ids'])
    print(f'{tokenizer.__class__.__name__}: {time.time() - tic:.6f} sec')
    return ret

def wwm_info(text, tokenizer):
    words = jieba.lcut(text)
    source, wwm = [], []
    for w in words:
        if len(w) == 1: ids = [tokenizer.convert_single_token(w)]
        else: ids = tokenizer.encode(w, add_special_tokens=False)
        source.extend(ids)
        wwm.extend([1]+[2]*(len(ids)-1))
    return source, wwm

if __name__ == '__main__':
    if 'make' in sys.argv: make_merged_vocab()
    import jieba_fast as jieba
    jieba.lcut('a')
    xtokenizer = CEBertTokenizer('vocab.txt')
    ftokenizer = CEBertTokenizerFast('vocab.txt')
    otokenizer = BertTokenizerFast('vocab.txt')
    print(len(xtokenizer.vocab))
    print(pre_tokenize('normalization 123abc 44apple'))
    z = '好SB啊！,normalization veryly goodness:   123abc每天…… 44appleless'
    #z = "The Low Memorial Library is a building at the center of Columbia University's Morningside Heights campus in Manhattan, New York City, United States. Designed by Charles Follen McKim of the firm McKim, Mead & White, the building was constructed between 1895 and 1897 as the central library of Columbia's library system. Columbia University president Seth Low funded the building and named it in memory of his father, Abiel Abbot Low. Its facade and interior are New York City designated landmarks, and the building is also a National Historic Landmark. Low is shaped like a Greek cross and is four stories tall, excluding a ground-level basement. The first floor contains an ambulatory around an octagonal rotunda. The stacks had space for 1.5 million volumes. The building was poorly suited for library use, but its central location made it a focal point of the university's campus. Following the completion of the much larger Butler Library in 1934, Low was converted to administrative offices."
    print(ftokenizer.convert_single_token(' '))
    print(xtokenizer.encode(z))
    print(xtokenizer.tokenize(z))
    print(ftokenizer.encode(z))
    print(ftokenizer.tokenize(z))
    print(ftokenizer.restore_token_list(z, ftokenizer.tokenize(z)))
    zz = ftokenizer.encode(z)
    xx = ftokenizer.convert_ids_to_tokens(zz)
    print(xx)
    print(ftokenizer.convert_tokens_to_string(xx))
    import ljqpy
    import numpy as np
    wikifn = '/mnt/data122/datasets/WikiCorpus/wiki.txt'
    wikigen = ljqpy.LoadListg(wikifn)
    wikis = [next(wikigen) for i in range(30)]
    if 'speed' in sys.argv:
        r1 = test_tokenize(xtokenizer, wikis)
        r2 = test_tokenize(ftokenizer, wikis)
        r3 = test_tokenize(otokenizer, wikis)
        print(len(r1), len(r2))
        for u, v in zip(r1, r2):
            assert np.array(u).sum() == np.array(v).sum()
    sys.exit()
    tokenizer = xtokenizer
    tic = time.time()
    #zz = lac.run(z)
    zz = jieba.lcut(z)
    print(zz)
    ss = []
    print(f'{time.time() - tic:.6f} sec')
    for w in zz:
        if len(w) == 1: ids = [tokenizer.convert_single_token(w)]
        else: ids = tokenizer.encode(w, add_special_tokens=False)
        ss.extend(ids)
    #zz = tokenizer.encode(z)
    print(f'{time.time() - tic:.6f} sec')
    #sys.exit()
    print('done')
    