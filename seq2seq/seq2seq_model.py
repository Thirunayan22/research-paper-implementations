import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import unicodedata
import matplotlib.pyplot as plt

import unidecode
import string
import time
import math
import re
import random
import matplotlib.ticker as ticker
import numpy as np
import os
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

plt.switch_backend("agg")

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
#LOADING DATA

# INDEX -> WORD and WORD -> INDEX CLASS WITH TOKENIZER

SOS_token =  0 #Start of sentence
EOS_token = 1

class Lang:
    def __init__(self,name):
        self.name=name
        self.word2index = {}
        self.index2word = {0:"SOS",1:"EOS"}
        self.word2count = {} # Dictionary containing number of times a word appears
        self.num_words = 2 # Already including <EOS> and <SOS> token

    #Adding values to global variables
    def add_word_to_vocab(self,word):
        if word not in self.word2index: #TODO
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self,sentence):
        for word in sentence.split(' '):
            self.add_word_to_vocab(word)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize("NFD",s) if unicodedata.category(c)!="Mn"
    )

#Lowercase, trim and remove non-character letter
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub("([.!?])",r" \1",s)
    s = re.sub("[^a-zA-z.!?]+", r" ",s)

    return s

def readLangs(language_1,language_2,reverse=False):

    lines = open("data/eng-fra.txt",encoding="utf-8").read().strip().split("\n") # list of individual word key pairs

    #splitting every line into pair
    pairs = [[normalizeString(s) for s in line.split("\t") ] for line in lines] # [["english","francois"],["english word","french word"],[...]...]

    if reverse==True:
        pairs = [list(reversed(pair))for pair in pairs]
        input_lang = Lang(language_2)
        output_lang = Lang(language_1)

    else:
        input_lang = Lang(language_1)
        output_lang = Lang(language_2)

    return input_lang,output_lang,pairs

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p)->bool:
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and p[1].startswith(eng_prefixes) #p[1] only works list is reversed i.e translating from French to English

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
#


def prepare_data(lang_1,lang_2,reverse=False):
    input_lang,output_lang,pairs = readLangs(lang_1,lang_2,reverse)
    print(f"Read {len(pairs)} sentence pairs")
    # print(pairs)
    print(pairs[100:200])
    pairs = filterPairs(pairs)
    print(f"Trimmed to {len(pairs)} filtered pairs")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    print(input_lang.name,input_lang.num_words)
    print(len(input_lang.word2index))
    print(output_lang.name,output_lang.num_words)
    return input_lang,output_lang,pairs

input_lang,output_lang,pairs = prepare_data("eng","fra",reverse=True)
# print(input_lang.word2index)

def showPlots(points):
    plt.figure()
    fig,ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


#ENCODER MODEL
# TOD Pass individual text into model
class EncoderModel(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(EncoderModel, self).__init__()
        self.hidden_size = hidden_size

        self.embedding  = nn.Embedding(input_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)

    def _init_hidden(self):
        hidden_layers = torch.zeros(1,1,self.hidden_size,device=device)
        return hidden_layers

    def forward(self,input,hidden):

        embedding_1 = self.embedding(input).view(1,1,-1)
        output = embedding_1
        output,hidden = self.gru(output,hidden)
        return output,hidden

class DecoderModel(nn.Module):
    def __init__(self,hidden_size,output_size):
        super(DecoderModel,self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size,hidden_size)
        self.GRU = nn.GRU(hidden_size,hidden_size)
        self.output_linear = nn.Linear(hidden_size,output_size)
        print("OUTPUT SIZE ",output_size )
        self.softmax =  nn.LogSoftmax(dim=1)

    def _init_hidden(self):
        return torch.zeros(1,1,self.hidden_size,device=device)

    def forward(self,input_x,hidden):
        output = self.embedding(input_x).view(1,1,-1)
        output = F.relu(output)
        output,hidden = self.GRU(output,hidden)
        # output = self.softmax(self.softmax(output[0])) # TODO CHANGED THIS TO THE BELOW ONE
        output = self.output_linear(output[0])
        output = self.softmax(output)

        return output,hidden

# BATCH_SIZE = 32
# EPOCHS = 500


def test_encoder(encoder):
    text_input = "I am Thirunayan Dinesh"


def indexesFromString(lang,sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang,sentence):
    indexes = indexesFromString(lang,sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes,dtype=torch.long,device=device).view(-1,1)

def tensorFromPairs(pair):
    input_tensors = tensorFromSentence(input_lang,pair[0])
    target_tensors = tensorFromSentence(output_lang,pair[1])
    return (input_tensors,target_tensors)


teacher_forcing_ratio  = 0.5
def train(input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion,max_length=MAX_LENGTH):

    encoder_hidden = encoder._init_hidden()
    print("ENCODER HIDDEN : " ,encoder_hidden)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length,encoder.hidden_size,device=device)
    loss = 0

    print(input_length)
    for encoder_input_idx in range(input_length):
        encoder_output,encoder_hidden = encoder(input_tensor[encoder_input_idx],encoder_hidden)
        encoder_outputs[encoder_input_idx] = encoder_output[0,0]


    decoder_input = torch.tensor([[SOS_token]],device=device,dtype=torch.long)
    decoder_hidden = encoder_hidden

    # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    print("TARGET LENGTH: ",target_length)
    print("TARGET TENSOR: ", len(target_tensor))
    use_teacher_forcing = True
    if use_teacher_forcing:
        i = 0
        for decoder_input_idx in range(target_length):
            decoder_output,decoder_hidden = decoder(decoder_input,decoder_hidden)
            i += 1
            print(f"REACHED POINT : {str(i)}")
            print("TARGET TENSOR: ",target_tensor[decoder_input_idx].size())
            decoder_input = target_tensor[decoder_input_idx]
            print("REACHED DECODER OUTPUT")
            print(f"DECODER OUTPUT : {decoder_output.size()}")
            loss+= criterion(decoder_output,target_tensor[decoder_input_idx])

    else:

        for decoder_input_idx in range(target_length):
            print("TARGET LENGTH: ", target_length)
            print("TARGET TENSOR: ", len(target_tensor))
            decoder_ouput,decoder_hidden = decoder(decoder_input,decoder_hidden)
            top_tensor,top_index = decoder_ouput.topk(1)
            decoder_input = top_index.squeeze().detach().long() #TODO MADE CHANGE changes top_tensor to top_index
            print(target_tensor[decoder_input_idx])
            loss += criterion(decoder_ouput,target_tensor[decoder_input_idx])
            if decoder_input.item() == EOS_token:
                break

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        return loss.item()/target_length


def asMinutes(s):
    m = math.floor(s/60)
    s -= m*60
    return  f"{m} {s}"

def timeSince(since,percent):
    now = time.time()
    s = now-since
    es = s/(percent)
    rs = es-s
    return f"{asMinutes(s)} - {asMinutes(rs)}"

def trainIters(encoder,decoder,epochs,print_every=1000,plot_every=1000,learning_rate=0.01):

    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(),lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(),lr=learning_rate)
    training_pairs = [tensorFromPairs(random.choice(pairs)) for i in range(epochs)] # selecting random pairs to translate
    criterion = nn.NLLLoss() #negative loss likelikelihood

    for iter in range(1,epochs+1):
        training_pair = training_pairs[iter-1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total/print_every
            print_loss_total = 0
            print("%s (%d %d) %.4f" % (timeSince(start,iter/epochs),epochs, iter/epochs*100,print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total/plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0



#running through a single sentence
def evaluate(encoder,decoder,sentence,max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang,sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder._init_hidden()

        #initializing encoder outputs so that they can be appended
        encoder_outputs = torch.zeros(max_length,encoder.hidden_size,device=device)
        print(f"ENCODER OUTPUT DIM : {encoder_outputs.ndimension}")

        #Loop through encoder inputs and feed them to encoder model
        for encoder_input_idx in range(input_length):
            encoder_output,encoder_hidden = encoder(input_tensor[encoder_input_idx],encoder_hidden)
            encoder_outputs[encoder_input_idx] += encoder_output[0,0]
            print(f"ENCODER OUTPUT {encoder_output}")

        decoder_input = torch.tensor([[SOS_token]],device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []


        for decoder_input in range(max_length):
            decoder_output,decoder_hidden = decoder(decoder_input,decoder_hidden,encoder_outputs)

            topv,topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item])

            decoder_input = topi.squeeze().detach()

        print("TOPI" ,topi)
        return decoded_words

def evaluateRandomly(encoder,decoder,n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print(">",pair[0])
        print("=",pair[1])

        output_words = evaluate(encoder,decoder,pair[0])
        output_sentence = ' '.join(output_words)
        print(f"OUTPUT SENTENCE: {output_sentence}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    hidden_size = 256
    encoder1 = EncoderModel(input_lang.num_words,hidden_size).to(device)
    decoder1 = DecoderModel(hidden_size=hidden_size,output_size=output_lang.num_words).to(device)

    trainIters(encoder1,decoder1,75000,print_every=5000)



#YESTERDAY NIGHT BUG STATUS
"""
The decoder output dimension should be [1,2803] but it was 256 , so what you did was you changed edited the decoder model
and it's output to be output = self.softmax(self.out(output[0])) where you had written  output = self.softmax(self.softmax(output[0]))

This was the correction you made\

Ongoing traceback error:
  File "seq2seq_model.py", line 362, in <module>
    trainIters(encoder1,decoder1,75000,print_every=5000)
  File "seq2seq_model.py", line 292, in trainIters
    print_loss_total += loss
TypeError: unsupported operand type(s) for +=: 'int' and 'NoneType'



"""




