import torch
import torch.nn as nn
from torchtext.legacy import data,datasets
from models import EncoderModel,AttentionDecoder


# TODO : CREATE INPUT USING TORCHTEXT
INPUT_LANGUAGE = data.Field()
TARGET_LANGUAGE = data.Field()

fields = {('input_language',INPUT_LANGUAGE),('target_language',TARGET_LANGUAGE)}


train_data,test_data = data.TabularDataset.splits(
    path='data',
    train='eng-fra.txt',
    test='test.txt',
    format= 'tsv',
    fields=fields,
    skip_header=True
)

INPUT_LANGUAGE.build_vocab(train_data.input_language)
TARGET_LANGUAGE.build_vocab(train_data.target_language)

print(f"VOCAB SIZE  : {len(INPUT_LANGUAGE.vocab)}")
print(f"VOCAB SIZE  : {len(TARGET_LANGUAGE.vocab)}")


print(vars(train_data[0]))

device = "cuda" if torch.cuda.is_available() else "cpu"

train_iterator,test_iterator = data.BucketIterator.splits(
    (train_data,test_data),
    sort=False,
    batch_size=32,
    device=device

)

i=0
for batch in train_iterator:
    print("=======================")
    print(f"BATCH - {i}")
    if i == 10:
        break
    else:
        print(batch)
        i+=1

    print("============================")

