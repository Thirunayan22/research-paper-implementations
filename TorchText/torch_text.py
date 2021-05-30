import torch
import torch.nn as nn
import torchtext
import torch.nn.functional as F
import torch.optim as optim
from torchtext.legacy import data
from torchtext.legacy import datasets
import random
import time
from tqdm import tqdm

SEED  = 1234 # SEED VALUE FIXED SO THAT WE GET THE SAME VALUE WHENEVER WE CALL A RANDOM NUMBER OR INITIALIZATION
torch.manual_seed(SEED)
# The Field defines how the data should be processed. In this sentiment analysis task the data consists of both the raw
# string of the review and the sentiment, either "positive" or "negative"

text = data.Field(tokenize="spacy",tokenizer_language="en_core_web_sm")
label = data.LabelField(dtype=torch.float)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Downloading data from IMDB dataset
train_data,test_data = datasets.IMDB.splits(text,label)

#Splitting train data into validation data
train_data,validation_data =  train_data.split(random_state=random.seed(SEED),split_ratio=0.8)

print("TRAIN DATA SAMPLE : " ,vars(train_data[0]))
print("VALIDATION DATA SAMPLE : " , vars(validation_data[0]))

print(f"Number of training examples : {len(train_data)}")
print(f"Number of testing examples : {len(test_data)}")
print(f"Number of validation examples : {len(validation_data)}")

# Next step is to build a vocabulary, when building a vocabulary  we change each word into a one-hot vector and assign
# indexes to each word

MAX_SIZE = 25000 #maximum number of words in sequence

text.build_vocab(train_data,max_size=MAX_SIZE)
label.build_vocab(train_data)

print(f"Unique words in TEXT vocabulary : {len(text.vocab)}")
print(f"Unique words in LABEL VOCAB : {len(label.vocab)}")

#We can view the most common words in the vocabulary :
print(text.vocab.freqs.most_common(20))

print("Word 'hello' to index : ",text.vocab.stoi["hello"]) # Test to index
print("Word at index 13000",text.vocab.itos[13000]) # Index to String


#Bucket Iterator class can be used to split data into batches
BATCH_SIZE = 64
device = torch.device(device)

train_iterator, validation_iteration,test_iterator = data.BucketIterator.splits(
    (train_data,validation_data,test_data),
    batch_size=BATCH_SIZE,
    device=device
)

# IN EACH BATCH, TEXT IS A TENSOR OF SIZE [SEQ_LENGTH,BATCH_SIZE].THAT IS A BATCH OF SENTENCES, EACH HAVING EACH WORD
# CONVERTED INTO A ONE-HOT VECTOR

#The input batch is then passed through the embedding layer to get embedded, which gives us a dense vector representation of our sentences.
# embedded is a tensor of size [sentence length, batch size, embedding dim].
#The RNN returns 2 tensors, output of size [sentence length, batch size, hidden dim] and hidden of size [1, batch size,
# hidden dim].
# output is the concatenation of the hidden state from every time step, whereas hidden is simply the final hidden state.
# We verify this using the assert statement. Note the squeeze method, which is used to remove a dimension of size 1.


class RNN(nn.Module):
    def __init__(self,input_dim,embedding_dim,hidden_dim,output_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_dim,embedding_dim)
        self.rnn = nn.RNN(embedding_dim,hidden_dim)
        self.final_linear =  nn.Linear(hidden_dim,output_dim)

    def forward(self,tokenized_input_sentence_batch):
        embedded = self.embedding(tokenized_input_sentence_batch)
        rnn_output, rnn_hidden = self.rnn(embedded)

        # rnn_output = [sent len, batch size, hid dim]
        # rnn_hidden = [1, batch size, hid dim]



        # rnn_output is the concatenation of every hidden state from every timestep, rnn_hidden is simply the final
        # hidden state. We can verify this using an assert function

        assert torch.equal(rnn_output[-1,:,:],rnn_hidden.squeeze(0))

        output = F.relu(self.final_linear(rnn_hidden.squeeze(0)))


        return output


# The input dimension is the dimension of the one-hot vectors, which is equal to the vocabulary size.

# The embedding dimension is the size of the dense word vectors.
# This is usually around 50-250 dimensions, but depends on the size of the vocabulary.

# The hidden dimension is the size of the hidden states. This is usually around 100-500 dimensions, but also depends
# on factors such as on the vocabulary size, the size of the dense vectors and the complexity of the task.

# The output dimension is usually the number of classes, however in the case of only 2 classes the output value is
# between 0 and 1 and thus can be 1-dimensional, i.e. a single scalar real number.

INPUT_DIM = len(text.vocab)
EMBEDDING_DIM = 100
OUTPUT_DIM = 1
HIDDEN_DIM = 256

model =  RNN(input_dim=INPUT_DIM,embedding_dim=EMBEDDING_DIM,output_dim=OUTPUT_DIM,hidden_dim=HIDDEN_DIM)

optimizer = optim.SGD(model.parameters(),lr=1e-3)
criterion = nn.BCEWithLogitsLoss()


#build function to count parameters in model
def count_parameters(model:nn.Module):

    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())

    print("TRAINABLE PARAMETERS : " ,trainable_parameters)
    print("NON TRAINABLE PARAMETERS : " ,non_trainable_parameters)
    print("TOTAL PARAMETERS  : ",total_parameters)
    return trainable_parameters,non_trainable_parameters,total_parameters


def binary_accuracy(preds,y):

    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum()/len(correct)
    return acc

count_parameters(model)

model.to(device)
criterion.to(device)


def train(model,iterator,optimizer,criterion):

    """
    The train function iterates over all examples, one batch at a time.

    model.train() is used to put the model in "training mode",
    which turns on dropout and batch normalization. Although we aren't using them in this model,
    it's good practice to include it. For each batch, we first zero the gradients. Each parameter in a model has a grad
    attribute which stores the gradient calculated by the criterion. PyTorch does not automatically remove (or "zero")
    the gradients calculated from the last gradient calculation, so they must be manually zeroed.

    We then feed the batch of sentences, batch.text, into the model. Note, you do not need to do
    model.forward(batch.text), simply calling the model works. The squeeze is needed as the predictions are initially
    size [batch size, 1], and we need to remove the dimension of size 1 as PyTorch expects the predictions input to
    our criterion function to be of size [batch size].

    The loss and accuracy are then calculated using our predictions and the labels, batch.label, with the loss being
    averaged over all examples in the batch.

    We calculate the gradient of each parameter with loss.backward(), and then update the parameters using the gradients
    and optimizer algorithm with optimizer.step().

    The loss and accuracy is accumulated across the epoch, the .item() method is used to extract a scalar from a tensor
    which only contains a single value.

    Finally, we return the loss and accuracy, averaged across the epoch. The len of an iterator is the number of
    batches in the iterator.

    You may recall when initializing the LABEL field, we set dtype=torch.float. This is because TorchText sets tensors
    to be LongTensors by default, however our criterion expects both inputs to be FloatTensors. Setting the dtype to be
    torch.float, did this for us. The alternative method of doing this would be to do the conversion inside the train
    function by passing batch.label.float() instad of batch.label to the criterion.

    """


    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)

        print(predictions.shape)

        loss = criterion(predictions,batch.label)
        acc = binary_accuracy(predictions,batch.label)

        print(len(batch.text))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        return epoch_loss/len(iterator) , epoch_acc/len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


N_EPOCHS = 5
best_valid_loss = float('inf')

for epoch in tqdm(range(N_EPOCHS)):

    start_time = time.time()

    train_loss,train_acc = train(model,train_iterator,optimizer,criterion)
    valid_loss,valid_acc = evaluate(model,validation_iteration,criterion=criterion)

    end_time = time.time()

    if valid_loss< best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(),'torch-text-model.pt')
















