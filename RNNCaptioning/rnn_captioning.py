import torch
import glob
import torch.nn as nn
from io import open
import os
import string
import unicodedata



all_letters =  string.ascii_letters+".,;'-"
n_letters = len(all_letters)+1 #EOS Marker String


#TODO Converting to ASCII to remove all special symbols
...

#TODO  Read a file and split into lines
...
#TODO  Build the category_lines dictionary, a list of lines per category
...

### TODO NETWORK DEFINITION

# TODO  One-hot vector for category
...

# TODO  One-hot matrix of first to last letters (not including EOS) for input
...

# TODO LongTensor of second letter to end (EOS) for target
...

# TODO Make category, input, and target tensors from a random category, line pair
...

# TODO TRAIN LOOP
...

# TODO TIME TRACKING
...

# TODO PLOTTING
...

# TODO Sample from a category and starting letter
...

# TODO Get multiple samples from one category and multiple starting letters
...
def unicodeToAscii(string):
    return ''.join(c for c in unicodedata.normalize("NFD",string))

#TODOCONVERT UNICODETOASCII