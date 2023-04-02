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

# Computes DICE loss
'''class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        smooth = 1.0
        y_true = F.interpolate(y_true, size=y_pred.size()[2:], mode='nearest')
        intersection = (y_pred * y_true).sum(dim=[2, 3])
        union = y_pred.sum(dim=[2, 3]) + y_true.sum(dim=[2, 3])
        dice = (2 * intersection + smooth) / (union + smooth)
        loss = 1 - dice.mean()
        return loss'''

def train(labelledset, valset, unlabelled_set, model, batch_size=16, epochs=100, k=0.1, loss_threshold=0.001, patience=10, baseline_full = False):
    def device():
        if torch.backends.mps.is_available():
            device = torch.device("mps") 
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        return device
    
    # Takes k most confident pseudolabels
    if baseline_full:
        k = 0
        k_pct = 'baseline_full'
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

    for epoch in range(epochs):
        
        if epoch == 0 or k==0:
            trainloader = DataLoader(labelledset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
        else:
            combinedset = ConcatDataset([labelledset, subset_unlabelled])
            trainloader = DataLoader(combinedset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
            subset_unlabelled = TensorDataset(torch.empty(0), torch.empty(0), torch.empty(0))

        print(f'Epoch {epoch+1}')
        model.train()
        train_loss = 0.0
        for images, masks, labels in tqdm(trainloader):
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(labelledset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks, labels in tqdm(valloader):
                images, masks, labels = images.to(device), masks.to(device), labels.to(device)
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

                # unlabelled subset
                subset_unlabelled = [TensorDataset(torch.empty(0), torch.empty(0), torch.empty(0))]
                confidences = torch.empty(0)
                for images, masks, labels in tqdm(unlabelled_loader):
                    images, masks, labels = images.to(device), masks.to(device), labels.to(device)
                    outputs = model(images)

                    unlabel_binary = torch.round(outputs)
                    confidence = torch.pow(torch.abs(outputs - unlabel_binary),2)
                    # Sum differences across all channels and pixels
                    confidence = confidence.view(confidence.size(0), -1).sum(dim=1)
                    confidences = torch.cat([confidences, confidence.cpu()])
                    '''
                    labelled_images = images.cpu()
                    labelled_masks = unlabel_binary.cpu()
                    labelled_labels = labels.cpu()

                    unlabelled_dataset = TensorDataset(labelled_images, labelled_masks, labelled_labels)
                    subset_unlabelled = ConcatDataset([subset_unlabelled, unlabelled_dataset])
                    
                    
                    confidence_threshold, confidence_threshold_ind = torch.min(top_confidences)
                
                    if confidence > confidence_threshold:
                        most_confident[confidence_threshold_ind] = confidence
                        
                    if k > images.size(0):
                        k = images.size(0)
                    ___, positions = torch.topk(confidence, k=k, largest=False)
                    labelled_images = images[positions].cpu()
                    labelled_masks = unlabel_binary.cpu()
                    labelled_labels = torch.ones(labelled_images.size(0), dtype=torch.long).cpu()
                    if labelled_images.size(0) > 0:
                        unlabelled_dataset = TensorDataset(labelled_images, labelled_masks, labelled_labels)
                        subset_unlabelled = ConcatDataset([subset_unlabelled, unlabelled_dataset])
                    else:
                        print('No new labelled images found')
                    '''
                confidences_argsort = torch.argsort(confidences)[:k]
                subset_unlabelled = Subset(unlabelled_set, confidences_argsort)

    

if __name__ == '__main__':
    trainset, valset, testset = load_dataset()
    trainset, valset, testset = trainset, valset, testset
    
    len_labelled = int(0.5 * len(trainset))
    len_unlabelled = len(trainset) - len_labelled
    labelledset, unlabelledset, valset, _= random_split(trainset, [len_labelled, len_unlabelled], generator=torch.Generator().manual_seed(42))
    train(trainset, valset, unlabelledset, model=Res_U_Net(), batch_size=4, epochs=10,k=0, loss_threshold=0.001, patience=10, baseline_full=True)
    
    for k in range(1,11):
        train(labelledset, valset, unlabelledset, model=Res_U_Net(), batch_size=4, epochs=10,k=k/10, loss_threshold=0.001, patience=5)
  