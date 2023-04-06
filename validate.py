import torch
from network import Res_U_Net
from load_dataset import load_dataset
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
from torchmetrics import Dice, JaccardIndex, Precision, Recall, F1Score
import csv
import os
#load in data

dire = os.path.dirname(os.path.abspath(__file__))


def device():
    if torch.backends.mps.is_available():
        device = torch.device("mps") 
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device



##function to calculate the dice score when given the predicted and ground truth masks
def dice_score(pred, target, device):
    """
    Calculates the Dice score between predicted and target binary segmentation masks.

    Args:
        pred (torch.Tensor): predicted segmentation mask with shape (batch_size, height, width)
        target (torch.Tensor): target segmentation mask with shape (batch_size, height, width)

    Returns:
        float: the Dice score between pred and target
    """
    dice = Dice(average='micro', num_classes=3).to(device)
    return dice(pred, target)

#function to calculate the iou score when given the predicted and ground truth masks
def iou_score(pred, target, device):
    """
    Calculates the IOU score (Jaccard Index) between predicted and target binary segmentation masks. Needs to be the same shape and order.

    Args:
        pred (torch.Tensor): predicted binary mask with shape (batch_size, height, width)
        target (torch.Tensor): target binary mask with shape (batch_size, height, width)

    Returns:
        float: the IOU score between pred and target given binary masks.

    """
    iou = JaccardIndex(task = 'multiclass', num_classes=3, average = 'micro').to(device)
    return iou(pred, target)

def precision_score(pred, target, device):
    """
    Calculates the Precision score between predicted and target binary segmentation masks.

    Args:
        pred (torch.Tensor): predicted segmentation mask with shape (batch_size, height, width)
        target (torch.Tensor): target segmentation mask with shape (batch_size, height, width)

    Returns:
        float: the Precision score between pred and target
    """
    precision = Precision(task = 'multiclass', num_classes=3, average='micro').to(device)
    return precision(pred, target)

def recall_score(pred, target, device):
    """
    Calculates the Recall score between predicted and target binary segmentation masks.

    Args:
        pred (torch.Tensor): predicted segmentation mask with shape (batch_size, height, width)
        target (torch.Tensor): target segmentation mask with shape (batch_size, height, width)

    Returns:
        float: the Recall score between pred and target
    """
    recall = Recall(task = 'multiclass', num_classes=3, average='micro').to(device)
    return recall(pred, target)

def f1_score(pred, target, device):
    """
    Calculates the F1 score between predicted and target binary segmentation masks.

    Args:
        pred (torch.Tensor): predicted segmentation mask with shape (batch_size, height, width)
        target (torch.Tensor): target segmentation mask with shape (batch_size, height, width)

    Returns:
        float: the F1 score between pred and target
    """
    f1 = F1Score(task = 'multiclass', num_classes=3, average='micro').to(device)
    return f1(pred, target)

def validate(device, pred,true_binary):
    """
    Validates the model by calculating the dice and iou scores for the predicted and ground truth masks.

    Args:
        device (torch.device): the device to be used for validation
        pred (torch.Tensor): predicted segmentation mask with shape (batch_size, height, width)
        target (torch.Tensor): target segmentation mask with shape (batch_size, height, width)

    Returns:
        float: the Dice score between pred and target
        float: the IOU score between pred and target given binary masks.
        float: the Precision score between pred and target
        float: the Recall score between pred and target
        float: the F1 score between pred and target

    """
    
    dice = dice_score(pred, true_binary, device)
    iou = iou_score(pred, true_binary, device)
    precision = precision_score(pred,true_binary, device)
    recall = recall_score(pred,true_binary, device)
    f1 = f1_score(pred,true_binary, device)
    return dice.item(), iou.item(), precision.item(), recall.item(), f1.item()

def metrics(model,device,testset):
    model.eval()
    model.to(device)
    testloader = DataLoader(testset, batch_size=16, num_workers=2, pin_memory=True)
    '''
    running_dice = 0.0
    running_iou = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_f1 = 0.0
    '''
    true=[]
    pred=[]
    with torch.no_grad():
        for i, (images, masks, labels )in enumerate(tqdm(testloader)):
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            y_true = masks
            y_pred = torch.argmax(model.forward(images),axis=1)
            true.extend(y_true.cpu().squeeze().tolist())
            pred.extend(y_pred.cpu().squeeze().tolist())
            '''
            running_dice += dice
            running_iou += iou
            running_precision += precision
            running_recall += recall
            running_f1 += f1
            '''
        true = torch.tensor(true).to(device)
        pred = torch.tensor(pred).to(device)
        dice, iou, precision, recall, f1 = validate(device,pred,true)
   
    # print("Dice: {} |  IOU: {} ".format(running_dice/len(testset),running_iou/len(testset)))
    return dice,iou, precision, recall, f1

def eval_model(model_path,device, testset):
    '''
    Evaluates the model on the test set and prints the dice and iou scores
    '''
    model = Res_U_Net()
    model.load_state_dict(torch.load(model_path))
    device = device()

    dice, iou, precision, recall, f1 = metrics(model,device,testset)
    print("{} |  Dice: {} |  IOU: {} | Precision: {} | Recall: {} | F1: {} ".format(model_path[:-3],dice,iou, precision, recall, f1))
    
    metrics_dict = {'Dice': dice, 'IOU': iou, 'Precision': precision, 'Recall': recall, 'F1': f1 }

    if not os.path.exists(dire + '/Results/'):
        os.makedirs(dire + "/Results")
        
    with open(dire + '/Results/' +model_path[:-3] + ".csv", "w", newline="") as fp:
                # Create a writer object
                writer = csv.DictWriter(fp, fieldnames=metrics_dict.keys())

                # Write the header row
                writer.writeheader()

                # Write the data rows
                writer.writerow(metrics_dict)
            
if __name__ == "__main__":
    model = Res_U_Net()
    #load in the model
    paths = ['model_k_0.0.pt','model_k_0.1.pt','model_k_0.2.pt','model_k_0.3.pt','model_k_0.4.pt','model_k_0.5.pt','model_k_0.6.pt','model_k_0.7.pt','model_k_0.8.pt','model_k_0.9.pt','model_k_1.0.pt','model_k_UB.pt']
    ##load in the data - testing batches

    trainset, valset, testset = load_dataset()
    for path in paths:
        eval_model(path,device,testset)




    #write in code example data (4d tensor of shape (batch_size, num_channels, height, width)) here for the validation functions

    # example_data = torch.rand(3,3,256,256)
    # example_data = example_data.type(torch.FloatTensor)

    # #write in code for the true binary mask with only values 0 or 1 (4d tensor of shape (batch_size, 1, height, width)) here for the validation functions
    # true_binary = torch.rand(3,1,256,256)
    # true_binary = true_binary.type(torch.FloatTensor)
    # true_binary = torch.round(true_binary)
    # device = device()
    # dice, iou = validate(model,device, example_data, true_binary)
    # print("Dice Score {:.3f} | IOU Score {:.3f}".format(dice, iou))

    

    





