import torch
import torch.nn as nn
import torch.optim as optim
import os


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout_probability):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.LSTM_1 = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)

        # *2 because LSTM is bidirectional and when passing to decoder
        # we want to map it from hidden_size*2 to hidden_size

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)

        # *2 because LSTM is bidirectional and when passing to decoder we want to map the cell states dimensions from
        # it from hidden_sisze*2 to hidden_size
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, input_x):
        output = self.dropout(self.embedding(input_x))
        encoder_states, (hidden, cell) = self.LSTM_1(output)

        print("hidden_state_size", hidden.size())
        print("cell_state_size", cell.size())

        hidden_fc = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]),
                                             dim=2))  # Concatenating hidden states from both directions across columns
        cell_fc = self.fc_cell(
            torch.cat((cell[0:1], cell[1:2]), dim=2))  # Concatenating cell states from both directions across columns

        return encoder_states, hidden_fc, cell_fc


##########################ATTENTION DECODER EXPLANATION ##################

"""

All the outputs states of the encoder LSTM denoted by encoder_states along with cell values and hidden_states are passed 
into the decoder.

Using these values the decoder creates a energy matrix which will then be input into the attention function. 
Using the equation : e = nn.Linear(previouse_decoder_hidden_at_i,encoder_hidden_state_at_i)
Then attention is calculated using : attention = softmax(e)
This  attention mask matrix is then concatenated with the outputs of the embedding layer in the decoder and then passed
to the LSTM layer of the decoder.

The outputs of this LSTM layers are then passed onto a fully connected layer which then does the final classification.   

"""

##########################ATTENTION DECODER EXPLANATION ##################


class AttentionDecoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,output_size,num_layers,dropout_probability):
        super(AttentionDecoder,self).__init__()

        self.embedding = nn.Embedding(input_size,embedding_size)
        self.dropout   = nn.Dropout(dropout_probability)

        self.LSTM = nn.LSTM(hidden_size*2+embedding_size,hidden_size,num_layers) # hidden_size*2 because the encoder_
        # outputs are bidirectional and + embedding_size, because we are concatenating the enocoder_steps with the embedding

        # here hidden_size is multiplied by 3 because we are first adding the encoder output and then we are also
        # inputting the previouse hidden_states from our decoder

        self.energy      = nn.Linear(hidden_size*3,1)
        self.FC_ouptput = nn.Linear(hidden_size,output_size)
        self.softmax = nn.Softmax(dim=0)

        self.relu = nn.ReLU()

    def forward(self,input_x:torch.Tensor,encoder_output_states:torch.Tensor,hidden_states:torch.Tensor,cell_states:torch.Tensor):

        """

        Encoder output states : (seq_length,N,hidden_size*2)

        Input x shape : (N) - we want it to be (1,N), seq_length is 1 here because we are sending in a single word and
        not a sentence

        """

        input_x = input_x.unsqueeze(0)
        print("INPUT SHAPE :"  ,input_x.size())

        embedding_output = self.dropout(self.embedding(input_x))
        #embedding shape : (1,N,embedding_size)

        sequence_length  = encoder_output_states.shape[0]

        # so that we can multiply the decoder  hidden_state with the encoder_states
        decoder_hidden_reshaped = hidden_states.repeat(sequence_length,1,1)
        energy = self.relu(self.energy(torch.cat((decoder_hidden_reshaped,encoder_output_states),dim=2)))
        attention = self.softmax(energy)

        # attention shape: (seq_length,N,1) where N is number of sequences

        attention = attention.permute(1,2,0)
        # transforming attention sh
        # ape to : (N,1,seq_length)
        encoder_output_states = encoder_output_states.permute(1,0,2)
        # make sure attention is in shape (N,1,hidden_size*2), so that we can multiply it with encoder_states

        # torch.bmm to perform 3-dimension multiplication : (attention * encoder_states)
        # context_vector shapes : (N,1,hidden_size*2) --> After permute --> (1,N,hidden_size*2)
        context_vector = torch.bmm(attention,encoder_output_states).permute(1,0,2)

        # concatenation of context_vector and embedding_output
        print("DECODER HIDDEN SHAPE : ",decoder_hidden_reshaped.size())
        print("ATTENTION SHAPE",attention.size())
        print("ENCODER STATES SHAPE : ",encoder_output_states.size())
        print("CONTEXT VECTOR SHAPE : " ,context_vector.size())
        print("EMBEDDING OUTPUT SHAPE : ",embedding_output.size())
        lstm_input = torch.cat((context_vector,embedding_output),dim=2)

        decoder_lstm_outputs, (hidden,cell) = self.LSTM(lstm_input,(hidden_states,cell_states))
        predictions = self.FC_ouptput(decoder_lstm_outputs)

        # predictions shape is : (1,N,target_vocab_size) but to send it to loss function we need it to be
        # (N,target_vocab_size), so we remove the first dimension by squeezing it

        predictions = predictions.squeeze(0)
        return predictions,hidden,cell

# TODO WHEN CALLING ENCODER WE HAVE TO PASS IN ENCODER HIDDEN LAYERS
#encoder_input _size
encoder_input_sample = torch.randint(low=0,high=100,size=(1,1))
decoder_input_sample = torch.tensor([3],dtype=
                                    torch.int64) #just index of one word because decoder receives one word at a time

# print(decoder_input_sample.size())

input_size_encoder = 10_000 # just a random vocab_size
input_size_decoder = 10_000 # encoder and decoder input sizes can be different

output_size = input_size_encoder #usually in translation tasks output size is the number of words in the target language

encoder_embedding_size = 300
decoder_embedding_size = 300

hidden_size = 1024 # LSTM and FCL (Fully Connected Layer) hidden size
num_layers = 1

encoder_model = Encoder(input_size=input_size_encoder,embedding_size=encoder_embedding_size,hidden_size=hidden_size,num_layers=num_layers,dropout_probability=0.2)

encoder_states,encoder_hidden_states,encoder_cell_states = encoder_model(encoder_input_sample)
print("ENCODER STATES SIZE : ", encoder_states.size())
print("HIDDEN STATES SIZE : " ,encoder_hidden_states.size())
print("CELL STATES SIZE : " ,encoder_cell_states.size())

#OUTPUTS
# ENCODER STATES SIZE :  torch.Size([1, 6, 2048])
# HIDDEN STATES SIZE :  torch.Size([1, 6, 1024])
# CELL STATES SIZE :  torch.Size([1, 6, 1024])

decoder_model = AttentionDecoder(input_size=input_size_decoder,embedding_size=decoder_embedding_size,
                                 hidden_size=hidden_size,num_layers=num_layers,
                                 output_size=output_size,dropout_probability=0.2)


predictions,decoder_hidden_states,decoder_cell_states = decoder_model(decoder_input_sample,encoder_states,encoder_hidden_states,encoder_cell_states)

print("DECODER PREDICTIONS SIZE " ,predictions.size())
print("DECODER HIDDEN STATES SIZE " ,decoder_hidden_states.size())
print("DECODER CELL STATES SIZE " ,decoder_cell_states.size())
print(torch.argmax(predictions))
