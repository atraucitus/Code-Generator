from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab as build_vocab_obj

SOS = "<sos>"
EOS = "<eos>"

def get_tokenizers():
    return get_tokenizer("spacy", language="en_core_web_sm"), get_tokenizer("spacy", language="en_core_web_sm")

def build_vocab(pairs, src_tokenizer, tgt_tokenizer, min_freq=2):
    counter = Counter()
    for src, tgt in pairs:
        counter.update([SOS] + src_tokenizer(src) + [EOS])
        counter.update([SOS] + tgt_tokenizer(tgt) + [EOS])

    v = build_vocab_obj(counter, specials=["<pad>", SOS, EOS], min_freq=min_freq)
    v.set_default_index(v["<pad>"])
    return v
