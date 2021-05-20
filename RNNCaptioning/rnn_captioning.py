import random

import torch
import glob
import torch.nn as nn
from io import open
import os
import string
import unidecode
from tqdm import tqdm

all_characters =  string.printable
n_letters = len(all_characters) + 1 #EOS Marker String
print(all_characters)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file = unidecode.unidecode(open("English.txt").read())

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(input_size,hidden_size)
        self.LSTM_1 = nn.LSTM(hidden_size,hidden_size,num_layers,batch_first=True) # Removing one dimension
        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self,input_x,hidden_state,cell_state):
        output = self.embed(input_x)
        output,(hidden,cell) = self.LSTM_1(output.unsqueeze(1),(hidden_state,cell_state))
        output = self.fc(output.reshape(output.shape[0],-1)) #Turning 2 dimensional matrix into 3 dimensional matrix
        return output, (hidden,cell)

    def init_hidden(self,batch_size):
        """
        Created hidden states and cell states
        """
        hidden = torch.zeros(self.num_layers,batch_size,self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers,batch_size,self.hidden_size).to(device)
        return hidden,cell

class Generator:
    """
    Generates Text Sequences
    """

    def __init__(self):
        self.chunk_len = 250 #Chunk of text
        self.num_epochs = 500 #number of epochs
        self.batch_size = 1
        self.print_every = 50 # Print results from training loop every 50 epochs
        self.hidden_size = 256
        self.num_layers = 2
        self.learning_rate = 0.003

    def char_tensor(self,string):

        """
        Generating list of indexes from given string
        """

        tensor = torch.zeros(len(string)).long()
        for character in range(len(string)):
            tensor[character] = all_characters.index(string[character])

        return tensor

    def get_random_batch(self):
        #TODO SEE TUTORIALS ON TENSOR RESHAPING AND TORCH TENSOR
        """
        Getting random batch from text file

        We use start_idx and end_idx to get random indexes and select a chunk of data from the text file

        Returns: Numerically indexed text_input and text_target
        """
        start_idx = random.randint(0,len(file)-self.chunk_len)
        end_idx = start_idx+self.chunk_len+1
        text_str = file[start_idx:end_idx]
        text_input = torch.zeros(self.batch_size,self.chunk_len) # Contains numerically indexed characters
        text_target = torch.zeros(self.batch_size,self.chunk_len) # Contains the character after the input character

        for i in range(self.batch_size): #Looping over batch size
            text_input[i,:] = self.char_tensor(text_str[:-1])
            text_target[i,:] = self.char_tensor(text_str[1:])

        return text_input.long(),text_target.long()


    def generate(self,initial_str="A",predict_len=100,temperature=0.85):
        #TODO LEARN ABOUT THIS METHOD ON VIDEO
        hidden,cell = self.rnn.init_hidden(batch_size=self.batch_size)
        initial_input = self.char_tensor(initial_str)

        predicted = initial_str

        for p in range(len(initial_str)-1):
            _,(hidden,cell) = self.rnn(
                initial_input[p].view(1).to(device),hidden,cell
            )

        last_char = initial_input[-1]
        for p in range(predict_len):
            output,(hidden,cell) = self.rnn(
                last_char.view(1).to(device),hidden,cell
            )

            output_dist = output.data.view(-1).div(temperature).exp()
            top_char = torch.multinomial(output_dist,1)[0]
            predicted_char = all_characters[top_char]
            predicted+= predicted_char
            last_char = self.char_tensor(predicted_char)
        return predicted

    def train(self):
        self.rnn = RNN(
            n_letters,self.hidden_size,self.num_layers,n_letters
        ).to(device)

        optimizer = torch.optim.Adam(self.rnn.parameters(),lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        print("=> Starting training")

        for epoch in tqdm(range(1,self.num_epochs+1)):
            print("EPOCH : ",epoch)
            inp,target = self.get_random_batch()
            hidden,cell = self.rnn.init_hidden(batch_size=self.batch_size)
            self.rnn.zero_grad()
            loss = 0
            inp = inp.to(device)
            target = target.to(device)

            for character_index in range(self.chunk_len):
                output,(hidden,cell) = self.rnn(inp[:,character_index],hidden,cell)
                loss += criterion(output,target[:,character_index])

            loss.backward()
            optimizer.step()
            loss = loss.item()/self.chunk_len

            if epoch % self.print_every ==0:
                print(f"Loss:{loss}")
                print(self.generate())


if __name__ == "__main__":
    gennames = Generator()
    gennames.train()




#TODO Converting to ASCII to remove all special symbols


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

#TODOCONVERT UNICODETOASCII