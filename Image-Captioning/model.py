import torch
import torch.nn as nn
import torchvision.models as models

"""
1) Encoder CNN
2) Decoder RNN
3) Converting CNN output to RNN
"""

class EncoderCNN(nn.Module):
    def __init__(self,embed_size,train_cnn=False):
        super(EncoderCNN, self).__init__()
        self.train_cnn = train_cnn

        self.inception_model = models.inception_v3(pretrained=True,aux_logits=False)
        self.inception_model.fc = nn.Linear(self.inception_model.fc.in_features,embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self,image):
        feature_embedding = self.inception_model(image)
        return self.dropout(self.relu(feature_embedding))

# EMBEDDING --> LSTM --> DENSE(LINEAR) LAYER
#DROPOUT FOR REGULARIZATION
class DecoderRNN(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.lstm  = nn.LSTM(embed_size,hidden_size,num_layers)
        self.linear  = nn.Linear(hidden_size,vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self,features,captions):
        embeddings = self.dropout(self.embedding(captions)) # TOKENIZED_CAPTIONS
        embeddings = torch.cat((features.unsqueeze(0),embeddings),dim=0) # incorporating features of image with embeddings of captions
        hidden, _ = self.lstm(embeddings)
        outputs = self.linear(hidden)
        return outputs

class CNNtoRNN(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size,hidden_size,vocab_size,num_layers)

    def forward(self,images,captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features,captions)
        return outputs

    def caption_image(self,image,vocabulary,max_lenght=50):
        result_caption = []
        with torch.no_grad():
            x = self.encoderCNN(image)
            states = None

            for _ in range(max_lenght):  # Generating Text to Label image
                hidden,states = self.decoderRNN.lstm_1(x,states)
                output = self.decoderRNN.linear(hidden.squeeze(0))
                predicted = output.argmax(1) # Getting predicted word with maximum probability
                result_caption.append(predicted.item())
                x = self.decoderRNN.embedding(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>": # Breaking loop if end of sentence is reached ::: itos is "item to string"
                    break

        return [vocabulary.itos[idx] for idx in result_caption]