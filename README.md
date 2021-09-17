# AIIJC. Трек "ИИ в медицине"

## Введение

"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."


## Данные

* [MosMedData: результаты исследований компьютерной томографии органов грудной клетки с признаками COVID-19]()  
* [MedSeg: COVID-19 CT segmentation dataset]()
* [Zenodo: COVID-19 CT Lung and Infection Segmentation Dataset](https://zenodo.org/record/3757476#.YRqU0IgzbP_)

## Запуск кода 

Для запуска обучения измените **config.py** и исполните `python3 main.py`

Для запуска сайта: `streamlit run app.py`

Для запуска предсказания: `python3 do_predictions.py --help`

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
