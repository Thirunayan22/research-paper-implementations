import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint,load_checkpoint,print_examples
from get_loader import get_loader
from model import CNNtoRNN

def train():
    transform = transforms.Compose([
        transforms.Resize((356,356)), # RESIZING IMAGE FOR FLICKR DATASET
        transforms.RandomCrop((299,299)), # DATA AUGMENTATION
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) ,#setting mean and standard deviation for all three channels

    ])

    train_loader,dataset = get_loader(
        root_folder = "flickr8k/images",
        annotation_file = "flickr8k/captions.txt",
        transform = transform,
        num_workers= 2,
    )
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = False
    train_CNN  = False

    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 1
    writer = SummaryWriter("runs/flickr")
    step = 0

    model =CNNtoRNN(embed_size,hidden_size,vocab_size,num_layers).to(device)
    criterion =nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"]) #making sure that Padding does not contribute to the error
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    for name,param in model.encoderCNN.inception_model.named_parameters():
        if "fc.weight" in name or "fc.bias" in name: # making sure gradient is only calculated for fully connected layers
            param.requires_grad = True

        else:
            param.required_grad = train_CNN

        if load_model:
            step = load_checkpoint(torch.load("my_chack.pth.tar"),model,optimizer)

        model.train()

    for epoch in range(num_epochs):
            print(epoch)
            if save_model:
                checkpoint = {
                    "state_dict":model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "step" : step
                }

                save_checkpoint(checkpoint)

            for idx,(imgs,captions) in tqdm(enumerate(train_loader),total=len(train_loader),leave=False):
                imgs = imgs.to(device)
                captions = captions.to(device)

                outputs = model(imgs,captions[:-1]) # getting captions
                loss = criterion(outputs.reshape(-1,outputs.shape[2]),captions.reshape(-1))
                writer.add_scalar("Training Loss",loss.item(),global_step=step)
                step += 1

                optimizer.zero_grad()
                loss.backward(loss)
                optimizer.step()

if __name__ == "__main__":
    train()


