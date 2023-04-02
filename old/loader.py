import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch
import torchvision
from PIL import Image
import tarfile
from sklearn.model_selection import train_test_split

def reading_tar_gz_files():
    """
    Extract the contents of two tar.gz files in the current directory.

    Returns:
        None
    """
    # Open the tar.gz file in read mode
    with tarfile.open('annotations.tar.gz', 'r:gz') as tar:

        # Extract all files to the current directory
        tar.extractall()

    with tarfile.open('images.tar.gz', 'r:gz') as tar:

        # Extract all files to the current directory
        tar.extractall()

def read_data_labels():
    """
    Read and concatenate two CSV files containing data labels.

    Returns:
        dataset (pd.DataFrame): A Pandas DataFrame containing the concatenated data.
    """

    # load the training set and the test set
    df_train = pd.read_csv(f"annotations/trainval.txt", sep=" ", names=["Image", "ID", "SPECIES", "BREED ID"])
    df_test = pd.read_csv(f"annotations/test.txt", sep=" ", names=["Image", "ID", "SPECIES", "BREED ID"])

    # we will concatenate it because the actual split is roughtly 50/50 and we might want another split ratio
    dataset = pd.concat([df_train, df_test])
    dataset.reset_index(drop=True)
    return dataset


# read + preprocessing
def load_data():
    """
    Load and preprocess the data.

    Returns:
        df (pd.DataFrame): A Pandas DataFrame containing the preprocessed data.
    """

    df = read_data_labels()
    df = preprocessing_data(df)
    return df

def create_class_mapping():
    """
    Create a mapping between class indices and class names.

    Returns:
        idx_to_class (dict): A dictionary that maps class indices to class names.
    """

    # Initialize empty lists for image IDs and labels
    image_ids = []
    labels = []

    # Open the annotation file and read image IDs and labels
    with open("annotations/trainval.txt", "r") as f:
        for line in f:
            line_parts = line.strip().split()
            image_id = line_parts[0]
            label = int(line_parts[1]) - 1
            image_ids.append(image_id)
            labels.append(label)

    # Create a set of unique image ID and label pairs
    unique_pairs = {(img_id.rsplit("_", 1)[0], label) for img_id, label in zip(image_ids, labels)}

    # Sort the unique pairs by label and join class names
    sorted_pairs = sorted(unique_pairs, key=lambda pair: pair[1])
    class_names = [" ".join(word.title() for word in pair[0].split("_")) for pair in sorted_pairs]

    # Create a dictionary mapping index to class name going from 1 to 37
    idx_to_class = {i: class_names[i-1] for i in range(1,len(class_names)+1)}

    return idx_to_class

def preprocessing_data(dataset):
    """
    Preprocess the data by mapping class IDs to class names.

    Args:
        dataset (pd.DataFrame): A Pandas DataFrame containing the data.

    Returns:
        dataset (pd.DataFrame): A Pandas DataFrame containing the preprocessed data.
    """
    idx_to_class = create_class_mapping()

    dataset['class'] = dataset['ID']
    for i in range(len(dataset)):
        dataset['class'].iloc[i] = idx_to_class[dataset['class'].iloc[i]]

    return dataset

def dataloading():
    """
    Load and split the Oxford-IIIT Pet dataset into training and validation sets.

    Returns:
    - train_dataset (OxfordPet): A PyTorch dataset containing the training images and their corresponding masks.
    - val_dataset (OxfordPet): A PyTorch dataset containing the validation images and their corresponding masks.
    """

    mask_path = 'annotations/trimaps/'
    # load the data
    data = load_data()

    # inputs/labels
    y = data['class']
    x = data['Image']

    trainval, x_test, y_trainval, y_test = train_test_split(x, y,stratify=y,test_size=0.2,random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(trainval, y_trainval,stratify=y_trainval,test_size=0.3,random_state=42)

    # why using this transformation
    img_transform = transforms.Compose([transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    label_transform = transforms.Compose([transforms.PILToTensor(),
                                       transforms.Resize((256, 256)),
                                       transforms.Lambda(lambda x: (x-1).squeeze().type(torch.LongTensor)) ])

    train_dataset = OxfordPet(x_train,mask_path, transform_img=img_transform, transform_label=label_transform)

    val_dataset = OxfordPet(x_val,mask_path, transform_img=img_transform, transform_label=label_transform)

    return train_dataset, val_dataset

class OxfordPet(Dataset):
    """
    A PyTorch dataset class for loading images and corresponding segmentation masks
    from the Oxford-IIIT Pet Dataset.

    Args:
        x (list): A list of image filenames.
        mask_path (str): Path to the directory containing segmentation masks.
        transform_img (torchvision.transforms.Compose): A set of image transformations.
        transform_label (torchvision.transforms.Compose): A set of label transformations.

    Attributes:
        x (list): A list of image filenames.
        len (int): The number of images in the dataset.
        mask_path (str): Path to the directory containing segmentation masks.
        transform_img (torchvision.transforms.Compose): A set of image transformations.
        transform_label (torchvision.transforms.Compose): A set of label transformations.

    Methods:
        __getitem__(self, index): Returns the transformed image and label for the given index.
        __len__(self): Returns the length of the dataset.
    """
    def __init__(self, x, mask_path, transform_img, transform_label):
        self.x = x
        self.len = len(x)
        # the mask files lies into 'annotations/trimaps'
        self.mask_path = mask_path
        self.transform_img = transform_img
        self.transfrom_label = transform_label

    def __getitem__(self, index):
        # a voir pour le +.jpg
        img = Image.open('images/' + self.x[index] + '.jpg')
        # the masks' name is quite different as it is a .png and not a .jpg
        mask = Image.open(self.mask_path+ self.x[index]+".png")
        # apply transformation on the img and the mask
        img_transformed= self.transform_img(img)
        mask_transformed = self.transfrom_label(mask)
        # return the transformed img/labels
        return img_transformed,mask_transformed

    def __len__(self):
        return self.len

if __name__=="__main__":
    train_set, val_set = dataloading()
