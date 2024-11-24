import os
import kagglehub
import torch
import joblib
from torchvision import models, transforms

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

    # Путь к скачанной модели
    model_path = f"{effnetb1full_pytorch_default_1_path}/modelefficientnetb10.884312_0.942337.pth"

    # Загружаем чекпоинт
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Удаляем префикс 'module.'
    cleaned_checkpoint = remove_module_prefix(checkpoint)

    # Сохраняем очищенный чекпоинт
    cleaned_model_path = "model"
    
    os.makedirs(cleaned_model_path, exist_ok=True)
    
    file_path = os.path.join(cleaned_model_path, 'efficientNet.pth')
    
    torch.save(cleaned_checkpoint, file_path)


def load_ResNet():
    resnet18full_pytorch_default_1_path = kagglehub.model_download('vasiliygorelov/resnet18full/PyTorch/default/1')

    # Path to the model file
    model_path = f"{resnet18full_pytorch_default_1_path}/modelresnetfull180.906556_0.952517.pth"

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Remove the 'module.' prefix from checkpoint keys
    cleaned_checkpoint = remove_module_prefix(checkpoint)
    
    # Сохраняем очищенный чекпоинт
    cleaned_model_path = "model"
    
    os.makedirs(cleaned_model_path, exist_ok=True)
    
    file_path = os.path.join(cleaned_model_path, 'ResNet.pth')
    
    torch.save(cleaned_checkpoint, file_path)


def load_autoML():
    automlnorm_other_default_1_path = kagglehub.model_download('vasiliygorelov/automlnorm/Other/default/1')

    # Construct the full path to the model file
    automl_model_path = f"{automlnorm_other_default_1_path}/automl_model_norm.pkl"


    # Создаём папку для сохранения модели, если её ещё нет
    save_dir = "model"
    os.makedirs(save_dir, exist_ok=True)

    # Копируем модель в папку models
    save_path = f"{save_dir}/automl_model_norm.pkl"
    joblib.dump(joblib.load(automl_model_path), save_path)
    
load_ResNet()
load_EfficientNet()
load_autoML()
