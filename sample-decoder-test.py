import torch
import torch.nn as nn

from models import *

def testAttentionDecoder():
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    embedding_size = 300
    hidden_size = 1024
    output_size = 10_000  # num of words
    max_len = 100
    decoder_input_sample = torch.tensor([[3]], device=device)  # replace in class with dictionary in the above method

    encoder_output_states, encoder_hidden = testEncoder()
    # attention_decoder_model = AttentionDecoder(embedding_size=embedding_size,
    #                                            hidden_size=hidden_size,output_size=output_size,
    #                                            dropout_probability=0.2,max_len=max_len
    #                                            ).to(device)

    attention_weights_model = AttnDecoderRNN(hidden_size=hidden_size, output_size=output_size, dropout_p=0.2,
                                             max_length=max_len, device=device)

    # decoder_hidden_input  = attention_decoder_model.initHidden()
    decoder_hidden_input = attention_weights_model.initHidden()
    print(decoder_hidden_input.device)
    #
    # attention_decoder_model_outputs = attention_decoder_model(input_x=decoder_input_sample,
    #                                                           encoder_output_states=encoder_output_states,
    #                                                           decoder_prev_hidden = decoder_hidden_input)

    decoder_hidden = attention_weights_model.initHidden()
    encoder_hidden_size = 100
    sample_encoder_outputs = torch.zeros(max_len, encoder_hidden_size, device=device)

    print("SAMPLE ENCODER OUTPUTS SIZE : ", sample_encoder_outputs.size())
    print("ACTUAL ENCODER OUTPUTS SIZE : ", encoder_output_states.size())

    attention_weights_hidden = attention_weights_model(input_x=decoder_input_sample, hidden_state=decoder_hidden,
                                                       encoder_outputs=sample_encoder_outputs)

    # print("ATTENTION DECODER MODEL OUTPUTS SHAPE : ",attention_decoder_model_outputs.shape)
    print("ATTENTION DECODER MODEL OUTPUTS SHAPE : ", attention_weights_hidden.shape)