import torch
from network import Res_U_Net
from load_dataset import load_dataset
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
from torchmetrics import Dice, JaccardIndex, Precision, Recall, F1Score
#load in data




def device():
    if torch.backends.mps.is_available():
        device = torch.device("mps") 
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device



##function to calculate the dice score when given the predicted and ground truth masks
def dice_score(pred, target):
    """
    Calculates the Dice score between predicted and target binary segmentation masks.

    Args:
        pred (torch.Tensor): predicted segmentation mask with shape (batch_size, height, width)
        target (torch.Tensor): target segmentation mask with shape (batch_size, height, width)

    Returns:
        float: the Dice score between pred and target
    """
    dice = Dice(average='weighted', num_classes=3)
    return dice(pred, target)

#function to calculate the iou score when given the predicted and ground truth masks
def iou_score(pred, target):
    """
    Calculates the IOU score (Jaccard Index) between predicted and target binary segmentation masks. Needs to be the same shape and order.

    Args:
        pred (torch.Tensor): predicted binary mask with shape (batch_size, height, width)
        target (torch.Tensor): target binary mask with shape (batch_size, height, width)

    Returns:
        float: the IOU score between pred and target given binary masks.

    """
    iou = JaccardIndex(task = 'multiclass', num_classes=3, average = 'weighted')
    return iou(pred, target)

def validate(model,device, samples,true_binary):
    """
    Validates the model by calculating the dice and iou scores for the predicted and ground truth masks.

    Args:
        model (torch.nn.Module): the model to be validated
        example_data (torch.Tensor): the example data to be used for validation
        device (torch.device): the device to be used for validation

    Returns:
        float: the dice score
        float: the iou score

    """
    
    model = model.to(device)
    samples = samples.to(device)
    pred = torch.argmax(model.forward(samples),axis=1)
    dice = dice_score(pred, true_binary)
    iou = iou_score(pred, true_binary)
    return dice.item(), iou.item()

def metrics(model,device,testset):
    model.eval()
    model.to(device)
    testloader = DataLoader(testset, batch_size=16, num_workers=2, pin_memory=True)
    running_dice = 0.0
    running_iou = 0.0
    with torch.no_grad():
        for images, masks, labels in tqdm(testloader):
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            dice,iou = validate(model,device,images,masks)
            running_dice += dice
            running_iou += iou
    # print("Dice: {} |  IOU: {} ".format(running_dice/len(testset),running_iou/len(testset)))
    return running_dice/len(testset),running_iou/len(testset)

def eval_model(model_path,device, testset):
    '''
    Evaluates the model on the test set and prints the dice and iou scores
    '''
    model = Res_U_Net()
    model.load_state_dict(torch.load(model_path))
    device = device()
    dice,iou = metrics(model,device,testset)
    print("{} |  Dice: {} |  IOU: {} ".format(model_path[:-3],dice,iou))


if __name__ == "__main__":
    model = Res_U_Net()
    #load in the model
    paths = ['model_k_0.1.pt','model_k_0.2.pt','model_k_0.3.pt','model_k_0.4.pt','model_k_0.5.pt','model_k_0.6.pt','model_k_0.7.pt','model_k_0.8.pt','model_k_0.9.pt','model_k_1.0.pt','model_k_baseline_full.pt']
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

    

    





