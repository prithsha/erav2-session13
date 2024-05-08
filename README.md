# Using Lightning module, RESNET18 and GradCam

Application uses Resnet18, PyTorch Lightning and Gradcam and show  misclassified images

## Image Augmentation, dataset, commonUtility and visualization
Application uses albumentations library for image augmentation

- utility\imageAugmentationUtility.py provided necessary function to provide image augmentation
- utility\imageAugmentationUtility.py provide necessary function to display one or more images. Function can accept tensor or nd-array to display image
- utility\cifar10Utility.py provide following dataset 
    - train
    - validation 
    - test
- utility\commonUtility.py provides function provides helper function to perform regular tasks like:
    - getting random dataset or images 
    - getting optimizer
    - checking matched and un-matched indexes
    - etc...

## Code execution flow

assignment_13_lightning.ipynb controls the execution 

1. cifar10 input is 32x32 image while resent 18 expects 224x224. So we have updated the starting layer of resnet18 . We are also using Resnet18 from torchvision.models.resnet18. 
2. We have created module LightResnet derived from LightningModule in models\modelHandler.py
    - Created three multiclass torch metrics for train, test and validation accuracy
    - Following functions Overrided. On each function we calculate accuracy
        - training_step
        - validation_step
        - test_step
        - configure_optimizers : for optimizer and scheduler

3. Using pytorch_lightning, Trainer class with following parameters 
    - logger: Using CSV logger which logs accuracy and learning parameter changes in csv format
    - Other logger option is to add TensorBoardLogger and CSVLogger as an array of loggers
    - Activate auto_lr_find = True so that we can find starting learning rate
    - TQDMProgressBar get used by default

4. Using trainer lr_find to find optimal learning rate to start with. You can plot and assign to LightningModule hyper params
5. Call trainer.fit , which executes train and validation function of model
6. Final test accuracy can be calculated using trainer.test
7. Show the accuracy and loss graph from saved metrics 
8. Save the model in "lightning_resnet18.pth"



## Results 

- After 30 epochs test accuracy was: 79.61 %
- For misclassified images and grdcam cam we have used saved model

