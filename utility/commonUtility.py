
import torch
from torch.utils.data import DataLoader 
import random
import torch.nn as nn
import torch.optim as optim
from torch_lr_finder import LRFinder


def get_random_images_batch_and_labels_from_data_loader(data_loader: DataLoader):
     
    selected_batch_index = random.randint(0, len(data_loader))  
    print(f"Total number of batches: {len(data_loader)}, batch_size : {data_loader.batch_size}), selected batch_index: {selected_batch_index}")
    selected_batch = None
    selected_batch_labels = None
    data_iterator = iter(data_loader)     
    i = 0 
    for value in data_iterator:
        if(i == selected_batch_index):
             selected_batch = value[0]
             selected_batch_labels = value[1]
        i = i + 1
    return selected_batch, selected_batch_labels


def get_images_at_specific_indexes_from_data_loader_batch(selected_batch, selected_batch_labels,selected_image_indexes):     
    images = []
    labels = []
    for i in selected_image_indexes:
            images.append(selected_batch[i])
            labels.append(selected_batch_labels[i])

    return images, labels     
     

def get_random_images_from_data_loader(data_loader: DataLoader, images_count = 10):
    selected_batch, selected_batch_labels = get_random_images_batch_and_labels_from_data_loader(data_loader)
    indices = list(range(len(selected_batch)))
    random.shuffle(indices)
    selected_image_indexes = indices[:images_count]

    images, labels = get_images_at_specific_indexes_from_data_loader_batch(selected_batch, selected_batch_labels,selected_image_indexes)
    return images, labels


def get_matched_and_non_matched_indices(predictions : torch.Tensor, ground_truth):
    comparison_result = predictions.eq(ground_truth)
    # Get the indices of True values
    matched_indices = torch.where(comparison_result)[0]
    non_matched_indices = torch.where(~comparison_result)[0]
    return matched_indices, non_matched_indices


def get_adam_optimizer(model: nn.Module, lr=0.05, weight_decay=1e-4) -> optim.Optimizer:
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer


def execute_model_on_batch(model, batch_images, batch_labels):
    
    output = model(batch_images)
    predictions = torch.argmax(output, dim=1)
    matched_indices, non_matched_indices = get_matched_and_non_matched_indices(predictions, batch_labels)
    print(f"matched_indices: {len(matched_indices)}")
    print(f"non_matched_indices: {len(non_matched_indices)}")
    return predictions, matched_indices, non_matched_indices


def get_images_for_matched_and_non_matched_model_predictions(model, batch_images, batch_labels, max_image_count = 10):
    
    predictions, matched_indices, non_matched_indices = execute_model_on_batch(model, batch_images, batch_labels)
    
    images = []
    predicted_labels = []
    actual_labels = []
    count = 0

    for i in non_matched_indices:
        images.append(batch_images[i])
        actual_labels.append(batch_labels[i])
        predicted_labels.append(predictions[i])
        if (count > 10):
            break
        count = count + 1
    non_matched_results = {"images" : images, "predicted_labels" : predicted_labels, "actual_labels" : actual_labels}


    images = []
    predicted_labels = []
    actual_labels = []
    count = 0
    for i in matched_indices:
        images.append(batch_images[i])
        actual_labels.append(batch_labels[i])
        predicted_labels.append(predictions[i])
        if (count > 10):
            break
        count = count + 1
    matched_results = {"images" : images, "predicted_labels" : predicted_labels, "actual_labels" : actual_labels}

    return non_matched_results, matched_results

def combine_labels(predictions, actuals):
    combined_labels = []
    for index in range(len(predictions)):
        combined_labels.append(f"{predictions[index]}/{actuals[index]}")
    return combined_labels





