import os
import glob

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms

import joblib
import pickle
import kagglehub

import streamlit as st 
import subprocess

from stl import mesh

from models import EfficientNetRegression

st.title('AI Integration')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[len('module.'):]  # Remove 'module.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

# Download the model EfficientNet
def load_EfficientNet():
    effnetb1full_pytorch_default_1_path = kagglehub.model_download('vasiliygorelov/effnetb1full/PyTorch/default/1')

    # Path to the downloaded model file
    model_path = f"{effnetb1full_pytorch_default_1_path}/modelefficientnetb10.884312_0.942337.pth"

    # Load the checkpoint
    checkpoint = torch.load(model_path)

    # Remove the 'module.' prefix from checkpoint keys
    cleaned_checkpoint = remove_module_prefix(checkpoint)

    model = EfficientNetRegression()
    model.load_state_dict(cleaned_checkpoint)
    return model.to(device)

def load_ResNet():
    resnet18full_pytorch_default_1_path = kagglehub.model_download('vasiliygorelov/resnet18full/PyTorch/default/1')

    # Path to the model file
    model_path = f"{resnet18full_pytorch_default_1_path}/modelresnetfull180.906556_0.952517.pth"

    # Load the checkpoint
    checkpoint2 = torch.load(model_path, map_location=torch.device('cpu'))

    # Remove the 'module.' prefix from checkpoint keys
    cleaned_checkpoint = remove_module_prefix(checkpoint2)

    model = models.resnet18(pretrained=True)

    # Modify the first convolutional layer to accept 6 channels
    # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # New: Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Get the original first conv layer
    original_conv = model.conv1

    # Create a new conv layer with 6 input channels
    new_conv = nn.Conv2d(9, original_conv.out_channels, kernel_size=original_conv.kernel_size,
                        stride=original_conv.stride, padding=original_conv.padding, bias=original_conv.bias)

    # Initialize the new conv layer's weights by copying the original weights and duplicating for the additional channels
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = original_conv.weight
        new_conv.weight[:, 3:6, :, :] = original_conv.weight  # Duplicate weights for the additional channels
        new_conv.weight[:, 6:9, :, :] = original_conv.weight
    # Replace the model's conv1 with the new conv layer
    model.conv1 = new_conv

    # Modify the fully connected layer for regression
    model.fc = nn.Linear(model.fc.in_features, 1)

    model.load_state_dict(cleaned_checkpoint)
    return model.to(device)

def load_autoML():
    automlnorm_other_default_1_path = kagglehub.model_download('vasiliygorelov/automlnorm/Other/default/1')

    # Construct the full path to the model file
    automl_model_path = f"{automlnorm_other_default_1_path}/automl_model_norm.pkl"

    # Load the AutoML model using joblib
    automl_model = joblib.load(automl_model_path)
    
def run_inference(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in dataloader:
            images = images.unsqueeze(0).to(device)
            
            outputs = model(images)
            predictions.extend(outputs.cpu().numpy().flatten())
    return predictions

model_efficientnet = load_EfficientNet()
model_resnet = load_ResNet()
model_autoML = load_autoML()

# Загрузка stl файла
st.subheader("load .stl file")

# Загружаем STL файл
stl_file = st.file_uploader("")

if stl_file is not None:
    # Сохраняем файл на диск
    input_dir = 'uploaded_files'
    
    os.makedirs(input_dir, exist_ok=True)
    file_path = os.path.join(input_dir, stl_file.name)

    with open(file_path, "wb") as f:
        f.write(stl_file.getbuffer())  # Записываем содержимое файла

    # Загружаем STL файл в объект mesh
    stl_mesh = mesh.Mesh.from_file(file_path)

    stl_path= os.path.join(input_dir, stl_file.name)
    
    stl_mesh.save(stl_path)
    
    st.write("Файл успешно загружен и сохранён!")
else:
    st.write("Пожалуйста, загрузите STL файл.")


with st.spinner('Creating projections, please wait...'):
    input_dir = 'uploaded_files'
    output_dir = 'images'
    
    os.makedirs(output_dir, exist_ok=True)

    # Аргументы для скрипты (директории и радиус)
    all_dirs = [input_dir, output_dir, '3']

    # Запускаем скрипт на с++ чтобы получить проекции
    subprocess.run(["./mesh_projection_mt", *all_dirs], capture_output=True)
    st.success('Ready!')


# Соберем проекции в один тензор
img_path = 'images'
images = []
for img in glob.glob(img_path + '/*.png'):
    if 'spherical' in img or 'cylinder' in img:
        images.append(Image.open(img).convert('RGB'))
    elif 'up' in img or 'left' in img or 'front' in img:
        images.append(Image.open(img).convert('L'))
    else: continue

if images is None:
    raise ValueError(f"Error loading image: {img_path}")

copy_images = images.copy()
# Выведем картинки
for img in copy_images:
    st.image(copy_images[0])

image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

images = [image_transforms(img) for img in images]

st.image(images[0])

input_tensor = torch.cat(images, dim = 0)

using_model = st.radio('Выберите модель: ', ('ResNet18', 'EffiecientNet', 'AutoML'))

outputs = 0.0
if using_model == 'ResNet18':
    model_resnet.eval()
    input_tensor = input_tensor.unsqueeze(0).to(device)
    outputs = model_resnet(images)
elif using_model == 'EffiecientNet':
    model_efficientnet.eval()
    input_tensor = input_tensor.unsqueeze(0).to(device)
    outputs = model_efficientnet(images)
elif using_model == 'AutoML':
    model_resnet.eval()
    model_efficientnet.eval()

    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    outputs_resnet = model_resnet(images)
    outputs_efficientNet = model_efficientnet(images)
    nn_outputs = pd.DataFrame({
        'resnet18_Prediction': outputs_resnet,
        'effnetb1full_Prediction': outputs_efficientNet
    })
    outputs = model_autoML.predict(nn_outputs).data
    
st.text(f'Полученный результат: {outputs}')
    
# Укажите путь к папке
folder_path = "путь_к_папке"


# Удалим файлы из папок:
for folder_path in ['uploaded_files', 'images']:
    # Убедитесь, что папка существует
    if os.path.exists(folder_path):
        # Перебираем файлы в папке
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Проверяем, что это файл (не папка)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Удаляем файл


