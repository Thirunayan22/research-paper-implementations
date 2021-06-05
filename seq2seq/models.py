import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Text,Dict,List

class EncoderModel(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(EncoderModel, self).__init__()
        self.hidden_size = hidden_size
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"

        self.embedding  = nn.Embedding(input_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)

    def _init_hidden(self):
        hidden_layers = torch.zeros(1,1,self.hidden_size,device=self.device)
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embedding = nn.Embedding(output_size,hidden_size)
        self.GRU = nn.GRU(hidden_size,hidden_size)
        self.output_linear = nn.Linear(hidden_size,output_size)
        # DEBUG print("OUTPUT SIZE ",output_size )
        self.softmax =  nn.LogSoftmax(dim=1)

    def _init_hidden(self):
        return torch.zeros(1,1,self.hidden_size,device=self.device)

    def forward(self,input_x,hidden):
        output = self.embedding(input_x).view(1,1,-1)
        output = F.relu(output)
        output,hidden = self.GRU(output,hidden)
        # output = self.softmax(self.softmax(output[0])) # TODO CHANGED THIS TO THE BELOW ONE
        output = self.output_linear(output[0])
        output = self.softmax(output)

        return output,hidden


class AttentionDecoder(nn.Module):
    def __init__(self,embedding_size:int,hidden_size:int,output_size:int,dropout_probability:float,max_len:int):
        super(AttentionDecoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.max_length  = max_len
        self.dropout_probability = dropout_probability

        # hidden_size+embedding_size because we would be concatentating previouse decoder's hidden
        self.embedding = nn.Embedding(output_size,embedding_size)
        self.attention_energy = nn.Linear(hidden_size+embedding_size,max_len)
        self.attention_combine = nn.Linear(hidden_size+embedding_size,hidden_size)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden_size,hidden_size)
        self.final_linear = nn.Linear(hidden_size,output_size)


        self.dropout = nn.Dropout(dropout_probability)

    def forward(self,input_x,encoder_output_states,decoder_prev_hidden):

        """

        Encoder output states : (seq_length,N,hidden_size*2)

        input_shape : (N) N is just the number of words usually 1
        """

        embedding_output = self.embedding(input_x).view(1,1,-1) # the dims of the embedding are matching the batch
        embedding_output = self.dropout(embedding_output)

        #embedding_output : 1x1xembedding_dim

        print("EMBEDDING OUTPUT DIM : " , embedding_output.shape)
        print("DECODER PREV HIDDEN : ",decoder_prev_hidden)

        attention_energy = self.attention_energy(torch.cat((embedding_output[0],decoder_prev_hidden[0]),dim=1))
        attention_weights = F.softmax(attention_energy,dim=1)

        attention_weights = attention_weights.unsqueeze(0)
        encoder_output_states = encoder_output_states.unsqueeze(0)

        print("ATTENTION RNN ATTENTION SHAPES : ", attention_weights.unsqueeze(0).shape)
        # DIMS :  1,1,100

        print("ENCODER OUTPUT SHAPES : ",encoder_output_states.unsqueeze(0).shape)
        # DIMS  :  1,100,1024


        # multiplying attention matrix with encoder_output
        # attention shape: (seq_length,N,1) where N is number of sequences

        attention_applied = torch.bmm(attention_weights,encoder_output_states)
        # ATTENTION APPLIED DIMS : (1,1,1024)

        #Concatenating applied attention weights and attention outputs
        attention_combine = torch.cat((embedding_output[0],attention_applied[0]),1)
        print(embedding_output[0].size())
        print(attention_applied[0].size())
        print(attention_combine.size())

        attention_combine = F.relu(self.attention_combine(attention_combine).unsqueeze(0))
        print("ATTENTION COMBINE",attention_combine.size())

        outputs , hidden = self.gru(attention_combine,decoder_prev_hidden)
        print("GRU OUTPUTS : " ,outputs.size())
        print("GRU HIDDEN : ",hidden.size())

        final_linear_output = F.log_softmax(self.final_linear(outputs))


        return final_linear_output,hidden,attention_weights
    def initHidden(self):
        device = "cpu"
        return torch.zeros(1, 1, self.hidden_size, device=device)

        #encoder_output shape : seq_lengthxNxhidden_dimx
        #torch.bmm


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p, max_length,device):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.device = device

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_x, hidden_state, encoder_outputs):
        embedded = self.embedding(input_x).view(1, 1, -1)
        embedded = self.dropout(embedded)


        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden_state[0]), 1)), dim=1)


        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))


        ## attn_applied_dimension  : 1,1,1024


        return attn_applied

    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size,device=self.device)

def testEncoder():

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(device)
    word2index = {
        "test_0":0 ,
        "test_1":1,
        "test_2":2
    }

    encoder_input_size = 10_000
    encoder_hidden_size = 1024
    encoder_input_sample = torch.tensor([word2index["test_0"]],device=device)

    encoder_model = EncoderModel(encoder_input_size,encoder_hidden_size).to(device)

    encoder_hidden_input = encoder_model._init_hidden()
    print(encoder_hidden_input.device)
    print(encoder_input_sample.device)

    encoder_output_states,encoder_model_hidden_states = encoder_model(encoder_input_sample,encoder_hidden_input)

    print("ENCODER MODEL OUTPUT SHAPE" , encoder_output_states.shape)
    print("ENCODER MODEL HIDDEN SHAPE", encoder_output_states.shape)
    print("ENCODER MODEL HIDDEN[0] SHAPE", encoder_output_states[0].shape)


    return encoder_output_states,encoder_model_hidden_states

def testAttentionDecoder():

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    embedding_size = 300
    decoder_hidden_size = 1024
    encoder_hidden_size = 1024
    output_size  = 10_000 #num of words
    max_len = 100
    decoder_input_sample = torch.tensor([[3]],device=device) #replace in class with dictionary in the above method

    encoder_output_states,encoder_hidden = testEncoder()
    attention_decoder_model = AttentionDecoder(embedding_size=embedding_size,
                                               hidden_size=decoder_hidden_size,output_size=output_size,
                                               dropout_probability=0.2,max_len=max_len
                                               ).to(device)


    decoder_hidden_input  = attention_decoder_model.initHidden()
    print(decoder_hidden_input.device)

    """
    In the below block we are creating a placeholder tensor filled with zeros with dimension 
    (max_len,encoder_hidden_size), and then we append the exact element from the encoder_output_states to this place 
    holder, this is because encoder output_states had dimension (1,1,encoder_hidden_size), which is incompatible when 
    sending into our decoder, to do batch matrix multiplication with the attention weights
    
    """


    encoder_outputs_placeholder = torch.zeros(max_len,encoder_hidden_size)
    encoder_outputs_placeholder[0]  = encoder_output_states[0,0]



    print("SAMPLE ENCODER OUTPUTS SIZE : ", encoder_outputs_placeholder.size())
    print("ACTUAL ENCODER OUTPUTS SIZE : ", encoder_output_states.size())

    decoder_outputs,decoder_hidden_states,attention_weights = attention_decoder_model(input_x=decoder_input_sample,
                                                              encoder_output_states=encoder_outputs_placeholder,
                                                              decoder_prev_hidden = decoder_hidden_input)




    # print("ATTENTION DECODER MODEL OUTPUTS SHAPE : ",attention_decoder_model_outputs.shape)
    print("ATTENTION DECODER MODEL OUTPUTS SHAPE : ",decoder_outputs.size())
    print("ATTENTION DECODER MODEL HIDDEN  SHAPE : ",decoder_hidden_states.size())
    print("ATTENTION DECODER MODEL ATTENTION WEIGHTS SHAPE : ",attention_weights.size())

    return decoder_outputs,decoder_hidden_states,attention_weights





