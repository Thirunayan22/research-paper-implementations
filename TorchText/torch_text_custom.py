import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data,datasets

#In this tutorial we load our own custom datasets from torchtext


NAME = data.Field()
SAYING = data.Field()
PLACE = data.Field()

fields = {'name': ('n', NAME), 'location': ('p', PLACE), 'quote': ('s', SAYING)}

train_data,test_data = data.TabularDataset.splits(
    path='json_data',
    train='train.json',
    test='test.json',
    format = 'json',
    fields=fields
    )

# x = vars(next(iter(train_data[0])))
# print(next(iter(train_data[0].s)))
print(vars(train_data[0]))

NAME.build_vocab(train_data)
SAYING.build_vocab(train_data)
PLACE.build_vocab(train_data)
BATCH_SIZE = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_iterator,test_iterator = data.BucketIterator.splits(
    (train_data,test_data),
    sort=False,
    batch_size=BATCH_SIZE,
    device=device
)

i = 0

for batch in train_iterator:
    print(f"BATCH: {i}")
    print(batch.n)
    i+=1

print(NAME.vocab.stoi['John'])


