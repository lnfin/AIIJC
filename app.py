import streamlit as st
import gdown
from PIL import Image
import numpy as np
import custom.models
from config import BinaryModelConfig, MultiModelConfig, LungsModelConfig
from utils import get_model
from production import get_predictions
from zipfile import ZipFile
import torch
import os
import base64
import cv2
from production import read_files, get_models

drive_link = 'https://drive.google.com/uc?id=1-tadxTBTRyru10rNNI0y4UcdntMK7hdh'


@st.cache
def download_model():
    gdown.cached_download(drive_link, quiet=False)


def main():
    st.markdown(
        f"""
    <style>
        .sidebar .sidebar-content {{
            background: url("https://i.ibb.co/BL3qFQW/background.png");
            background-repeat: repeat;
            background-size: 100% auto;
    }}
        .reportview-container {{
            background: url("https://i.ibb.co/BL3qFQW/background.png");
            background-repeat: repeat;
            background-size: 100% auto;
        }}
        .reportview-container .main .block-container{{
            max-width: 850px;
            padding-top: 0rem;
            padding-right: 0rem;
            padding-left: 0rem;
            padding-bottom: 0rem;
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )
    # download_model()
    for folder in ['segmentations/', 'images/']:
        if not os.path.exists(folder):
            os.mkdir(folder)

    st.title('Сегментация поражения легких коронавирусной пневмонией')

    st.subheader("Загрузка файлов")
    filenames = st.file_uploader('Выберите или ператащите сюда снимки', type=['png', 'jpeg', 'jpg', '.nii', '.nii.gz'],
                                 accept_multiple_files=True)

    multi_class = st.checkbox(label='Мульти-классовая сегментация', value=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    binary_model, multi_model, lungs_model = get_models()
    binary_model = binary_model.to(device)
    multi_model = multi_model.to(device)
    lungs_model = lungs_model.to(device)

    if st.button('Загрузить') and filenames:
        print(filenames)
        images, folder_name = read_files(filenames)

        model = None
        cfg = BinaryModelConfig
        if multi_class:
            model = multi_model
            cfg = MultiModelConfig

        if not images:
            st.error('Неправильный формат или название файла')
        else:
            user_dir = "segmentations/" + folder_name
            os.mkdir(user_dir)

            zip_obj = ZipFile(user_dir + 'segmentations.zip', 'w')
            with st.expander("Информация о каждом фото"):
                info = st.info('Делаем предсказания, пожалуйста, подождите')
                for image_list in images:
                    for filename, pred in zip(image_list[:2],
                                              get_predictions(cfg, binary_model, lungs_model, image_list[:2], device,
                                                              multi_model=model)):
                        info.empty()
                        st.markdown(f'<h3>{filename.split("/")[-1]}</h3>', unsafe_allow_html=True)

                        original = np.array(Image.open(filename))
                        col1, col2 = st.columns(2)
                        col1.header("Оригинал")
                        col1.image(original, width=350)

                        to_img = pred * 255
                        image_path = user_dir + filename.split('/')[-1]
                        cv2.imwrite(image_path, to_img)
                        zip_obj.write(image_path)

                        col2.header("Сегментация")
                        col2.image(pred, width=350)

                        st.markdown('<br />', unsafe_allow_html=True)

                zip_obj.close()

            with st.expander("Скачать сегментации"):
                with open('segmentations.zip', 'rb') as file:
                    st.download_button(
                        label="Архив сегментаций",
                        data=file,
                        file_name="segmentations.zip",
                    )

            for file in os.listdir(user_dir):
                os.remove(user_dir + file)

            image_dir = 'images/' + folder_name
            for file in os.listdir(image_dir):
                os.remove(image_dir + file)

            os.rmdir(user_dir)
            os.rmdir(image_dir)


if __name__ == '__main__':
    main()
