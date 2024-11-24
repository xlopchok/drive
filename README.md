# Решение команды AI Integration 

# Локальная установка:
1. Клонируйте репозиторий
2. Выполните следующие команды
```
docker build -t app 'AI PO'
docker run -p 8501:8501 --gpus all app 
