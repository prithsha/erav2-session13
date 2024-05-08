import gradio as gr
import torch
from models import modelHandler
from utility import cifar10Utility
from torchvision import transforms
from PIL import Image
import numpy as np

MODEL_NAME = "lightning_resnet18.pth"
BATCH_SIZE = 512

_model = None

def get_model():
    model_handler = modelHandler.ModelHandler(batch_size=BATCH_SIZE)
    new_loaded_model = model_handler.get_lightning_model_instance(saved_model=MODEL_NAME)
    model_handler.show_model_summary(new_loaded_model)
    # Set the model to evaluation mode (disable dropout, randomness, etc.)
    new_loaded_model = new_loaded_model.eval()
    return new_loaded_model

def predict_image(model, selected_image, use_gradcam = False):
    predictions = model(selected_image)
    predicted_label = cifar10Utility.get_labels_names(torch.argmax(predictions).item())[0]

    top_3_predictions = torch.argsort(predictions, descending = True)[0][0:3]
    top_3_predictions = list(top_3_predictions.detach().numpy())
    top_3_prediction_labels = cifar10Utility.get_labels_names(top_3_predictions)

    if(use_gradcam):
        print("Gradcam is asked")

    pil_image = None
    if isinstance(selected_image, torch.Tensor):
        image_array = (selected_image.squeeze().permute(1, 2, 0) * 255).byte().numpy()
        pil_image = Image.fromarray(image_array)

    return predicted_label, top_3_prediction_labels, pil_image

def _create_image_to_tensor(img : Image):
    print(f"old image size: {img.size}")
    img = img.convert('RGB')
    img = img.resize((32, 32))
    image_tensor = transforms.ToTensor()(img).unsqueeze(0) 
    print(f"Image tensor shape: {image_tensor.shape}")
    return image_tensor

def read_image(image_path):
    img = Image.open(image_path)
    return _create_image_to_tensor(img)

def perform_action(image_input, use_gradcam, selected_layer):

    global _model

    if(_model is None):
        _model = get_model()

    tensor_image = _create_image_to_tensor(image_input)
    predicted_label, top_3_prediction_labels, gadcam_image = predict_image(_model, tensor_image, use_gradcam=use_gradcam)

    return predicted_label, [top_3_prediction_labels], gadcam_image, gadcam_image

output_text = gr.Textbox(label="Prediction output")
output_List = gr.Dataframe(label="Top three predictions",col_count=3)
grad_cam_image = gr.Image( label="Gradcam output", width=64)
layer_output_image = gr.Image(label="Output of model selected layer", width=64)

app = gr.Interface(
    fn=perform_action,
    inputs=[gr.Image(type="pil"), gr.Checkbox(value=True, label="Use Gradcam"), gr.Dropdown(choices=["layer-1", "layer-2", "layer-3", "layer-4"], value="layer-4", label="Select Resnet8 layer output")],
    
    outputs= [output_text , output_List, grad_cam_image, layer_output_image]
)

app.launch()

