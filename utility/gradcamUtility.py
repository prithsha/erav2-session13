from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def create_grad_cam_overlaid_images(model, target_layers, images, predictions_labels, actual_labels, image_weight = 0.5):
    cam = GradCAM(model=model, target_layers=target_layers)

    heatmap_overlaid_images = []
    for index in range(len(actual_labels)):
        input_tensor = images[index].unsqueeze(dim=0)
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(int(predictions_labels[index]))], aug_smooth=True)
        grayscale_cam = grayscale_cam[0, :]
        final_image = input_tensor.numpy() / 255
        final_image = final_image.squeeze(0).transpose((1, 2, 0))
        heat_map = show_cam_on_image(final_image , grayscale_cam, use_rgb=True, image_weight=image_weight)
        heatmap_overlaid_images.append(heat_map)
    
    return heatmap_overlaid_images