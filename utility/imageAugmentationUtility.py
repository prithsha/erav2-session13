import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_cifar10_train_and_test_transforms(image_set_mean :list, image_set_std : list):
     
    train_transforms_collection = []
    train_transforms_collection.extend([A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),     
                                        A.RandomCrop(height=32, width=32, always_apply=True),
                                        A.Resize(height=32, width=32, always_apply=True)])   
    

    train_transforms_collection.append(A.CoarseDropout(max_holes=2, min_holes=1, 
                                        max_height=16, max_width=16,
                                        min_height=4, min_width=4,
                                        p=0.25, fill_value=[0, 255, 255]))
    
    train_transforms_collection.extend([ A.Normalize(mean=tuple(image_set_mean), std=tuple(image_set_std)),
                                                ToTensorV2()])

    train_transforms = A.Compose(train_transforms_collection)
    test_transforms = A.Compose([ A.Normalize(mean=tuple(image_set_mean), std=tuple(image_set_std)),  ToTensorV2()])

    return train_transforms, test_transforms