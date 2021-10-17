# AIIJC. Трек "ИИ в медицине"

## Данные

* [MosMedData: результаты исследований компьютерной томографии органов грудной клетки с признаками COVID-19](https://mosmed.ai/datasets/covid19_1110/)  
* [MedSeg: COVID-19 CT segmentation dataset](http://medicalsegmentation.com/covid19/)
* [Zenodo: COVID-19 CT Lung and Infection Segmentation Dataset](https://zenodo.org/record/3757476#.YRqU0IgzbP_)

## Запуск предсказания

Установка зависимостей: `pip3 install -r requirements.txt`

Для удобного предсказания для жюри мы сделали **command.py**:

Аргументы:
- **--data**: папка или архив rar/zip с dicom файлами
- **--save_folder**: папка куда сохранить предсказанные сегментации
- **--multi**: указывать этот флаг, если нужно предсказание разных видов повреждения

## Запуска сайта

Установка зависимостей: `pip3 install -r requirements.txt`

Для запуска сайта: `streamlit run app.py`

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


Папку c весами моделей скачать [тут](https://drive.google.com/drive/folders/1rbbjqClLp1PCjHGqBdXnAopvVtrc4ipL?usp=sharing)

[Тренировка](https://colab.research.google.com/drive/1t-RMsJp1dJuR12D5CHJh2cfQ6gjH-a95?usp=sharing) происходила на Google Colab Pro
