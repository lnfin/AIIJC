# AIIJC. Трек "ИИ в медицине"

## Данные

* [MosMedData: результаты исследований компьютерной томографии органов грудной клетки с признаками COVID-19](https://mosmed.ai/datasets/covid19_1110/)  
* [MedSeg: COVID-19 CT segmentation dataset](http://medicalsegmentation.com/covid19/)
* [Zenodo: COVID-19 CT Lung and Infection Segmentation Dataset](https://zenodo.org/record/3757476#.YRqU0IgzbP_)

## Запуск кода 

Для запуска сайта: `streamlit run app.py`

Для запуска предсказания: `python3 inference.py -h`

## Структура проекта
- **app.py** - код сайта
- **main.py** - главный модуль, запускается для тренировки
- **config.py** - файл конфигураций
- **data_functions.py** - модули подготовки данных для обучения
- **train_functions.py** - модули тренировки и валидации
- **productions.py** - методы подготовки тестовых данных для предсказания
- **utils.py** - вспомогательные модули
- **custom** - содержит кастомные модули
  - losses.py - кастомные функции потерь 
  - metrics.py - кастомные метрики
  - models.py - кастомные классы моделей
- **requirements.txt** - файл зависимостей
- **checkpoints**
  - Binary.pth
  - MultiClass.pth
  - Lungs.pth

Папку checkpoints скачать [тут](https://drive.google.com/file/d/19svztOBB4RhnW7cwuZTDPZb0EiWKdydN/view?usp=sharing)

[Тренировка](https://drive.google.com/drive/folders/1AawFssvVRtF3lZIr6nOvObcH41Gu4rIE?usp=sharing) происходила на Google Colab Pro
