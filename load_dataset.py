import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data import TensorDataset, ConcatDataset, random_split


def load_dataset():
    # Define the root directory where the dataset should be stored
    print('Loading dataset...')
    root = ''

    # Load the dataset using the OxfordIIITPet class with download=True
    # do not apply transforms here as they affect the loading of the targets
    train_val_data = datasets.OxfordIIITPet(root=root, split='trainval', 
                                            target_types=['category','segmentation'], download=True)
    test_data = datasets.OxfordIIITPet(root=root, split='test',
                                target_types=['category','segmentation'], download=True)

    dataset = ConcatDataset([train_val_data, test_data])

    # Define the transform to apply to images
    img_transform = transforms.Compose([transforms.Resize((224, 224)),  # resize the images to 224x224 pixels
                                        transforms.ToTensor()  # convert the images to tensors, apply scaling (from 0-255 to 0-1)
                                    ])

    # Define the transform to apply to masks
    mask_transform = transforms.Compose([transforms.Resize((224, 224)),  # resize the images to 224x224 pixels
                                        transforms.PILToTensor(),       # convert to tensor, do not apply scaling
                                        transforms.Lambda(lambda x: x -1) # remove 1 since pixel classes are 1-indexed
                                        ])

    # loop through all images, apply transforms and store in lists
    # cannot directly apply transforms due to (class, mask) tuple in original dataset
    all_img = []
    all_mask = []
    all_label = []

    for i, datapoint in enumerate(dataset):
        img, targets = datapoint
        class_label, mask = targets
        
        # apply transforms to image
        img = img_transform(img)
        all_img.append(img)
        # apply transforms to mask
        mask = mask_transform(mask)
        all_mask.append(mask)
        # apply transforms to label
        all_label.append(class_label)
        
    # create new dataset
    dataset = TensorDataset(torch.stack(all_img),torch.stack(all_mask),torch.tensor(all_label))

    # create train, val, test splits (70%,10%,20%)
    len_train = int(0.7 * len(dataset))
    len_val = int(0.1 * len(dataset))
    len_test = len(dataset) - len_train - len_val
    trainset, valset, testset = random_split(dataset, [len_train, len_val, len_test], generator=torch.Generator().manual_seed(42))
    print('Dataset loaded.')
    return trainset, valset, testset
