import os
import glob
import logging
import shutil

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

logger = logging.getLogger("MyLogger")
logger.setLevel(logging.DEBUG)

st.markdown(
    """
    <style>
    .stApp {
        background-color: rgb(0, 0, 0); 
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='color: rgb(60,255,2);'>AI Integration</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='color: rgbrgb(255,35,166);'>AI Integration</h2>", unsafe_allow_html=True)

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
    # Указываем путь к модели внутри контейнера
    model_path = "model/efficientNet.pth"

    # Загружаем чекпоинт модели
    checkpoint = torch.load(model_path)

    # Инициализируем модель
    model = EfficientNetRegression()
    model.load_state_dict(checkpoint)

    # Переносим модель на устройство (GPU/CPU)
    return model.to(device)

def load_ResNet():
    # Указываем путь к модели внутри контейнера
    model_path = "model/ResNet.pth"

    # Загружаем чекпоинт модели
    checkpoint = torch.load(model_path)

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

    model.load_state_dict(checkpoint)
    return model.to(device)

def load_autoML():
    # Путь к модели внутри контейнера
    automl_model_path = "model/automl_model_norm.pkl"

    # Загружаем модель
    automl_model = joblib.load(automl_model_path)
    return automl_model
    
def run_inference(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in dataloader:
            images = images.unsqueeze(0).to(device)
            
            outputs = model(images)
            predictions.extend(outputs.cpu().numpy().flatten())
    return predictions

with st.spinner('Загрузка модели, подождите...'):
    model_efficientnet = load_EfficientNet()
    model_resnet = load_ResNet()
    model_autoML = load_autoML()

# Загрузка stl файла
st.subheader("Загрузите stl файл")

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
    img_path = os.path.join(img_path, stl_file.name[:-4])
    images = []
    
    output_log_file = "variables_log.txt"

    with open(output_log_file, "w") as log_file:
        log_file.write(f"img_path: {img_path}\n")
        log_file.write(f"Image Files: {glob.glob(img_path + '/*.png')}\n")

    for img in glob.glob(img_path + '/*.png'):
        if 'spherical' in img or 'cylinder' in img:
            images.append(Image.open(img).convert('RGB'))
        elif 'up' in img or 'left' in img or 'front' in img:
            images.append(Image.open(img).convert('L'))
        else: continue

    if images is None:
        raise ValueError(f"Error loading image: {img_path}")

    # Выведем картинки
    st.image(images)

    image_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    images = [image_transforms(img) for img in images]

    input_tensor = torch.cat(images, dim = 0)

    using_model = st.radio('Выберите модель: ', ('ResNet18', 'EffiecientNet', 'AutoML'))

    outputs = 0.0
    if using_model == 'ResNet18':
        model_resnet.eval()
        input_tensor = input_tensor.unsqueeze(0).to(device)
        outputs = model_resnet(input_tensor).squeeze().item()
    elif using_model == 'EffiecientNet':
        model_efficientnet.eval()
        input_tensor = input_tensor.unsqueeze(0).to(device)
        outputs = model_efficientnet(input_tensor).squeeze().item()
    elif using_model == 'AutoML':
        model_resnet.eval()
        model_efficientnet.eval()

        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        outputs_resnet = model_resnet(input_tensor).squeeze().item()
        outputs_efficientNet = model_efficientnet(input_tensor).squeeze().item()
        nn_outputs = pd.DataFrame({
            'resnet18_Prediction': [outputs_resnet],
            'effnetb1full_Prediction': [outputs_efficientNet]
        })
        outputs = model_autoML.predict(nn_outputs).data.squeeze().item()
        
    st.text(f'Полученный результат: {outputs}')

    # # Удалим файлы из папок:
    # for folder_path in ['uploaded_files', 'images']:
    #     # Убедитесь, что папка существует
    #     if os.path.exists(folder_path):
    #         # Перебираем файлы в папке
    #         for file_name in os.listdir(folder_path):
    #             file_path = os.path.join(folder_path, file_name)

    #             # Проверяем, что это файл (не папка)
    #             if os.path.isfile(file_path):
    #                 os.remove(file_path)  # Удаляем файл
                    
    # Удалим файлы из папок:
    for dir_path in ['uploaded_files', 'images']:
        # Удаляем все содержимое папки images
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)  # Удаление файла
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Удаление папки и её содержимого


