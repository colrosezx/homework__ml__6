# Transformer

Этот проект представляет собой минимальную, но функциональную реализацию архитектуры **Transformer** на PyTorch с применением к задаче машинного перевода. Поддерживается **mixed precision training** для ускорения и экономии памяти, а также используется внешний токенизатор (по умолчанию — Mistral).


## Структура проекта

transformer_translation/
├── train.py # Скрипт для обучения (1 итерация)
├── model.py # Модель Transformer
├── tokenizer.py # Работа с токенизатором
├── checkpoints/ # Сохранённые веса модели
└── logs/ # JSON-файл с логом обучения
---

## Запуск обучения

### 1. Установка зависимостей

```bash
pip install torch tokenizers transformers
```
2. Запуск
```bash
python train.py
```
После запуска:
- Весы модели сохранятся в: checkpoints/transformer.pt (Отсутствует, тк слишком большой файл для гитхаба)
- Лог обучения будет доступен в: logs/train_log.json