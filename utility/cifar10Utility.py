
import torch
import torch.utils
import torch.utils.data
from torchvision import datasets
from torch.utils.data import DataLoader 

class CustomCIFAR10(datasets.CIFAR10):

    def apply_augmentations(self, image):
        augmented_image = self.transform(image=image)  # Pass 'image' as named argument
        return augmented_image

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed["image"]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_datasets(train_transforms_collection, test_transforms_collection, data_folder) -> tuple[datasets.CIFAR10, datasets.CIFAR10]:

    train_dataset = CustomCIFAR10( root=data_folder, train=True, download=True, transform=train_transforms_collection)    
    test_dataset = CustomCIFAR10( root=data_folder, train=False, download=True, transform=test_transforms_collection)

    # use 20% of training data for validation
    train_set_size = int(len(train_dataset) * 0.8)
    valid_set_size = len(train_dataset) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [train_set_size, valid_set_size], generator=seed)
    
    return train_dataset, validation_dataset, test_dataset


def get_dataloaders(train_dataset, test_dataset, validation_dataset = None, batch_size = 128, shuffle=True, num_workers=4, pin_memory=True) -> tuple[DataLoader, DataLoader]:
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
    validation_loader = None
    if(validation_dataset is not None):
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)

    for batch_data, label in test_dataloader:
    # (e.g., shape: (batch_size, 1 channel, 28, 28)). (batch_size, channels, height, width)
    # y would contain the corresponding labels for each image, indicating the actual digit represented in the image 
        print(f"Shape of test_dataloader batch_data [Batch, C, H, W]: {batch_data.shape}")
        print(f"Shape of test_dataloader label (label): {label.shape} {label.dtype}")
        # print(f"Labels for a batch of size {batch_size} are {label}")
        break

    return train_dataloader, validation_loader, test_dataloader


def get_mean():
    return [x / 255.0 for x in [125.3, 123.0, 113.9]]

def get_std():
    return [x / 255.0 for x in [63.0, 62.1, 66.7]]

def get_image_classes():
    return ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def get_labels_names(labels_indexes) -> list:

    if not isinstance(labels_indexes, list):
        labels_indexes = [labels_indexes]

    labels = []
    image_classes = list(get_image_classes())
    for index in labels_indexes:
            labels.append(image_classes[index])
    return labels