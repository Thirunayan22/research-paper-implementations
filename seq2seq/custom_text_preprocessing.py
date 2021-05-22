import torch
import torch.nn as nn
import re
from unidecode import unidecode
import os
import unicodedata


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

def sentenceToTensor(sentence):
    for word in sentence:
        pass

# def apply_filter(max_len=MAX_LENGHT,pair):
#
#     if senten


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
    return torch.tensor(sentence_tensor,dtype=torch.long,device=device)

def tensorFromPairs(input_lang,target_lang,pair):

    input_tensor = tensorFromSentence(input_lang,pair[0])
    target_tensor = tensorFromSentence(target_lang,pair[1])

    return (input_tensor,target_tensor)




dataset_path = f"data/eng-fra.txt"
input_language,target_language,pairs = read_text("eng","fra",max_length=MAX_LENGHT,dataset_path=dataset_path,reverse=True)

print("Sentence :",pairs[0] )
print("Tensor : " , tensorFromPairs(target_lang=target_language,input_lang=input_language,pair=pairs[0]))#OUTPUT LANGUAGE IS ENGLISH
print("INPUT LANGUAGE",input_language.lang_name)
print("OUTPUT LANGUAGE",target_language.lang_name)
# print("SAMPLE PAIRS ",pairs)

# def train(input_tensor,target)