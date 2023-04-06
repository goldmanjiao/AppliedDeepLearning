import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data import TensorDataset, ConcatDataset, random_split,DataLoader, Subset
from load_dataset import load_dataset
from validate import validate
from network import Res_U_Net
import numpy as np
from torch import nn, optim
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models


def train(labelledset, valset, unlabelled_set, model, batch_size=16, epochs=100, k=0.1, loss_threshold=0.001, patience=10, UB = False):

    """
    This function trains a given variant of the model

     Args:
        labelledset: Segment of the training set with groundtruth labels
        valset: labelled validation set
        unlabelledset: Segment of the training set with unlabelled examples
        model: Network architecture to train
        batch_size: examples per minibatch
        epochs: number of epochs
        k: threshold percentage for proportion pseudolabelled examples to feed back into training set
        loss_threshold: threshold for early stopping, with early stopping if validation loss > best_val_loss - loss_threshold
        patience: number of epochs beyond which early stopping can occur
        UB: if true, trains on entire labelled training set

    Returns:
        None
    
    """

    def device():
        if torch.backends.mps.is_available():
            device = torch.device("mps") 
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        return device
    
    # Takes k most confident pseudolabels
    if UB:
        k = 0
        k_pct = 'UB'
    else:
        k_pct = k
        k = int(len(unlabelled_set)*k)
    
    device = device()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    early_stop_counter = 0

    unlabelled_loader = DataLoader(unlabelled_set, batch_size=batch_size, num_workers=0, pin_memory=False, shuffle=False)
    valloader = DataLoader(valset, batch_size=batch_size, num_workers=0, pin_memory=True, drop_last=True)

    # Trains model on both labelled set and top k% unlabelled examples
    for epoch in range(epochs):
        
        if epoch == 0 or k==0:
            trainloader = DataLoader(labelledset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
        else:
            combinedset = ConcatDataset([labelledset, subset_unlabelled])
            trainloader = DataLoader(combinedset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
            subset_unlabelled = [TensorDataset(torch.empty(0), torch.empty(0), torch.empty(0))]

        print(f'Epoch {epoch+1}')
        model.train()
        train_loss = 0.0
        for images, masks, labels in tqdm(trainloader):
            images, masks, labels = images.to(device), masks.type(torch.LongTensor).to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(labelledset)

        model.eval()

        # Computes loss on validation set for early stopping
        val_loss = 0.0
        with torch.no_grad():
            for images, masks, labels in tqdm(valloader):
                images, masks, labels = images.to(device), masks.type(torch.LongTensor).to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
            val_loss /= len(valset)
  
        if val_loss < best_val_loss - loss_threshold:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), 'model_k_' + str(k_pct) + '.pt')
            print(f'Saved model at epoch {epoch+1}')
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                return model

        print(f'Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}')
        #model.load_state_dict(torch.load('model.pt'))
        model.eval()
        if k > 0:
            with torch.no_grad():
                criterion_pseudo = nn.CrossEntropyLoss(reduction='none') 

                # Computes trimap segmentation for all unlabelled images, as well as the confidence of each segmentation mask
                subset_unlabelled = [TensorDataset(torch.empty(0), torch.empty(0), torch.empty(0))]
                confidences = torch.empty(0)
                for images, masks, labels in tqdm(unlabelled_loader):
                    images, masks, labels = images.to(device), masks.to(device), labels.to(device)
                    outputs = model(images)

                    unlabel_binary_indices = torch.argmax(outputs, dim=1)
                    confidence = criterion_pseudo(outputs, unlabel_binary_indices)
         
                    confidence = confidence.view(confidence.size(0), -1).sum(dim=1)
                    confidences = torch.cat([confidences, confidence.cpu()])
                
                # Ranks pseudolabels for all unlabelled examples and selects top k percent
                confidences_argsort = torch.argsort(confidences, descending=True)[:k]
                subset_unlabelled = Subset(unlabelled_set, confidences_argsort)
                

    

if __name__ == '__main__':
    trainset, valset, testset = load_dataset()
    trainset, valset, testset = trainset, valset, testset
    
    len_labelled = int(0.5 * len(trainset))

    len_unlabelled = len(trainset) - len_labelled
    labelledset, unlabelledset= random_split(trainset, [len_labelled, len_unlabelled], generator=torch.Generator().manual_seed(42))
    train(trainset, valset, unlabelledset, model=Res_U_Net(), batch_size=4, epochs=10,k=0, loss_threshold=0.001, patience=10, UB=True)

    for k in range(0,11):
        train(labelledset, valset, unlabelledset, model=Res_U_Net(), batch_size=4, epochs=10,k=k/10, loss_threshold=0.001, patience=5)
        if k>0:
            continue
