import random

import torch
import torch.nn as nn
import re
from unidecode import unidecode
import os
import unicodedata
from seq2seq_model import EncoderModel,DecoderModel
import time
import torch.optim as optim
from typing import List
from tqdm import tqdm

MAX_LENGHT = 10
SOS_TOKEN = 0
EOS_TOKEN = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
class PreprocessSeq2SeqInput:
    def __init__(self,language,end_of_sentence_tag="<EOS>",start_of_sentence_tag="<SOS>"):
        self.lang_name = language
        self.word2index = {}
        self.index2word = {0:start_of_sentence_tag,1:end_of_sentence_tag}
        self.word2count = {}
        self.num_words = 2

    #use regex to remove non character letters

    def addWord2Vocab(self,word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] =1
            self.index2word[self.num_words] = word
            self.num_words+=1

        else:
            self.word2count[word]+=1

    def addSentenceToVocab(self,sentence):

        for word in sentence.split(" "):
            self.addWord2Vocab(word)

def unicodeToASCII(text):

    return ''.join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def normalizeString(s):
    s = unicodeToASCII(s.lower().strip())
    s = re.sub("([.!?])", r" \1", s)
    s = re.sub("[^a-zA-z.!?]+", r" ", s)
    return s


eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def read_text(language_1,language_2,dataset_path,max_length,reverse=False):

    with open(dataset_path,"r",encoding="utf-8") as dataset:
        lang_dataset = dataset.read().strip().split("\n")

    pairs = [[normalizeString(english_sentence) for english_sentence in line.split("\t")] for line in lang_dataset]
    print(pairs[:10])
    apply_filter = lambda pair: len(pair[0].split(' ')) < max_length and len(pair[1].split(' '))< max_length and pair[1].startswith(eng_prefixes)

    if reverse==True: #If you want to translate from french to english

        pairs  = [list(reversed(pair)) for pair in pairs]
        pairs = [pair for pair in pairs if apply_filter(pair) == True]

        input_language = PreprocessSeq2SeqInput(language_2)
        target_language = PreprocessSeq2SeqInput(language_1)

    else:
        input_language = PreprocessSeq2SeqInput(language_1)
        target_language = PreprocessSeq2SeqInput(language_2)

    for pair in pairs:
        input_language.addSentenceToVocab(pair[0])
        target_language.addSentenceToVocab(pair[1])

    print(pairs[:10])

    return input_language,target_language,pairs

def tensorFromSentence(lang,sentence):

    sentence_tensor = [lang.word2index[word] for word in sentence.split(' ')]
    sentence_tensor.append(EOS_TOKEN)
    return torch.tensor(sentence_tensor,dtype=torch.long,device=device).view(-1,1) #adding an extra dimension on column

def tensorFromPairs(input_lang,target_lang,pair):

    input_tensor = tensorFromSentence(input_lang,pair[0])
    target_tensor = tensorFromSentence(target_lang,pair[1])

    return (input_tensor,target_tensor)

def train(input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion,max_length=MAX_LENGHT):

    encoder_hidden = encoder._init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length,encoder.hidden_size,device=device)
    loss = 0

    for encoder_input_word_idx in range(input_length):
        encoder_output,encoder_hidden = encoder(input_tensor[encoder_input_word_idx],encoder_hidden)
        encoder_outputs[encoder_input_word_idx] = encoder_output[0,0] #appending hidden state value for each word in sentence

    decoder_input = torch.tensor([[SOS_TOKEN]],dtype=torch.long,device=device)
    decoder_hidden = encoder_hidden

    # use_teacher_forcing = True if random.random() > 0.5 else False
    use_teacher_forcing = True

    if use_teacher_forcing:
        for decoder_input_idx in range(target_length):

            decoder_output,decoder_hidden = decoder(decoder_input,decoder_hidden)
            decoder_input = target_tensor[decoder_input_idx]

            # print("DECODER OUTPUT : ",decoder_output.size())
            # print("TARGET SIZE :  ",target_tensor[decoder_input_idx].size())
            # print("TARGET VALUE : " ,target_tensor[decoder_input_idx])
            loss += criterion(decoder_output,target_tensor[decoder_input_idx]) # We are able insert the decoder output directly with the target tensor output becuase we are using negative log loss likelihood loss

    else:
        for decoder_input_idx in range(target_length):
            decoder_output,decoder_hidden = decoder(decoder_input,decoder_hidden)
            top_tensor,top_index = decoder_output.topk(1)
            decoder_input = top_index.squeeze().detach().long()  # returns index of word with highest probability for decoder output
            # print("DECODER OUTPUT : ",decoder_output.size())
            # print("TARGET :  ",target_tensor[decoder_input_idx].size())
            loss += criterion(decoder_output,target_tensor[decoder_input_idx]) #teacher-forcing==false loss
            if decoder_input.item() == EOS_TOKEN: #CHECKING IF INDEX IS 1 WHICH IS THE EOS TOKEN
                break


    loss.backward() # backprop
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item()/target_length

def trainIters(input_lang,target_lang,encoder:EncoderModel,decoder:DecoderModel,pairs:List,epochs:int,print_every=1000,plot_every=1000,learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(),lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(),lr=learning_rate)

    training_pairs = [tensorFromPairs(input_lang=input_lang,target_lang=target_lang,pair=random.choice(pairs)) for i in range(epochs)]
    criterion = nn.NLLLoss() # Negative log loss likelihood

    print("Training PAIRS",training_pairs[:10])

    for iter in tqdm(range(1,epochs+1)): #EPOCHS+1 SO THAT  EPOCHS DOESN'T START AT ZERO AND PREVIOUS RANGE(EPOCHS) IS NOT ZERO
        training_pair = training_pairs[iter-1] #iter-1 because training pair indexing starts at 0
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        # print("INPUT TENSOR IN TRAIN ITER : " ,input_tensor)
        # print("TARGET TENSOR IN TRAIN ITER : ",target_tensor)

        loss = train(input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion)
        print_loss_total += loss


def save_models(encoder_model,decoder_model,save_folder):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)


    torch.save(encoder_model,f"{save_folder}/custom_encoder_model.pth")
    torch.save(decoder_model,f"{save_folder}/custom_decoder_model.pth")




if __name__ == "__main__":

    dataset_path = f"data/eng-fra.txt"
    input_language,target_language,pairs = read_text("eng","fra",max_length=MAX_LENGHT,dataset_path=dataset_path,reverse=True)

    print("Sentence :",pairs[0] )
    print("Tensor : " , tensorFromPairs(target_lang=target_language,input_lang=input_language,pair=pairs[0]))#OUTPUT LANGUAGE IS ENGLISH
    print("INPUT LANGUAGE",input_language.lang_name)
    print("OUTPUT LANGUAGE",target_language.lang_name)
    # print("SAMPLE PAIRS ",pairs)
    input_sample_tensor,target_sample_tensor = tensorFromPairs(input_lang=input_language,target_lang=target_language,pair=pairs[0])
    print(len(pairs))


    encoder_model = torch.load("encoder_model.pth")
    decoder_model = torch.load("decoder_model.pth")


    ############### ENCODER TESTING #############################################################
    encoder_input = input_sample_tensor[0]
    encoder_outputs = torch.zeros(MAX_LENGHT,encoder_model.hidden_size,device=device)
    encoder_hidden  = encoder_model._init_hidden()

    # print(input_sample_tensor.shape)
    print("ENCODER INPUT : ",encoder_input)
    output,hidden = encoder_model(encoder_input,encoder_hidden) #insert word indexes one by one
    print("ENCODER OUTPUT : ", output[0,0].shape)
    print("ENCODER HIDDEN ", hidden.shape)
    ############### ENCODER TESTING #############################################################

    ############### DECODER TESTING #############################################################


    decoder_input = torch.tensor([[SOS_TOKEN]],device=device,dtype=torch.long)
    decoder_ouptputs = torch.zeros(input_language.num_words)
    decoder_hidden = decoder_model._init_hidden()

    print("DECODER INPUT : ",decoder_input)
    decoder_output,decoder_hidden = decoder_model(decoder_input,decoder_hidden)

    print("DECODER OUTPUT : " ,decoder_output.topk(1)[1].squeeze().detach().long().item()) # returns index of word with highest probability for decoder output
    print("DECODER HIDDEN : " ,decoder_hidden.shape)

    ############### DECODER TESTING #############################################################


    print("STARTING MODEL TRAINING.....")
    trainIters(input_lang=input_language,target_lang=target_language,encoder=encoder_model,decoder=decoder_model,pairs=pairs,epochs=10000)

    save_models(encoder_model,decoder_model,"./models")