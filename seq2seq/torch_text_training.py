import torch
import torch.nn as nn
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.data import Field,TabularDataset,BucketIterator
import spacy

spacy_en = spacy.load("en_core_web_sm")

def spacy_tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


quote = Field(sequential=True,use_vocab=True,tokenize=spacy_tokenizer,lower=True)
label = Field(sequential=False,use_vocab=False)

fields = {"quote":("q",quote),"score":("s",label)}

train_data,test_data = TabularDataset.splits(
    path="json_data",train="train.json",test="test.json",format="json",fields=fields
)


quote.build_vocab(train_data,max_size=10000,min_freq=1)

print(next(iter(quote.vocab)))
train_iterator,test_iterator = BucketIterator.splits(
    (train_data,test_data),batch_size=2,device="cpu"
)

print(next(iter(train_iterator)))

for batch in train_iterator:
    print(batch)
    print(batch.dataset)
# print(data)

print(quote.vocab.stoi["potato"])