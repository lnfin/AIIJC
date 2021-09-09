import streamlit as st
import gdown
from PIL import Image
import numpy as np
import custom.models
from config import Cfg
from production import get_predictions
from zipfile import ZipFile
import random
import string
import os
import base64
import cv2

drive_link = 'https://drive.google.com/uc?id=1-tadxTBTRyru10rNNI0y4UcdntMK7hdh'


@st.cache
def download_model():
    gdown.cached_download(drive_link, quiet=False)


@st.cache
def read_files(files):
    imgs = list()
    for file in files:
        with open('images/' + file.name, 'wb') as f:
            f.write(file.getvalue())

        imgs.append('images/' + file.name)
    return imgs


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
    if not os.path.exists('segmentations/'):
        os.mkdir('segmentations/')

    st.title('Сегментация поражения легких коронавирусной пневмонией')

    st.subheader("Загрузка файлов")
    filenames = st.file_uploader('Выберите или ператащите сюда снимки', type=['png', 'jpeg', 'jpg'],
                                 accept_multiple_files=True)

    multi_class = st.checkbox(label='Мульти-классовая сегментация', value=True)

    if st.button('Загрузить') and filenames:
        print(filenames)
        images = read_files(filenames)
        print(len(images))
        
        user_dir = f"segmentations/{''.join(random.choice(string.ascii_lowercase) for _ in range(7))}/"
        os.mkdir(user_dir)
        
        cfg = Cfg(multi_class)
        zip_obj = ZipFile('segmentations.zip', 'w')
        with st.expander("Информация о каждом фото"):
            info = st.info('Делаем предсказания, пожалуйста, подождите')
            for filename, pred in zip(images, get_predictions(cfg, images)):    
                info.empty()
                st.markdown(f'<h3>{filename}</h3>', unsafe_allow_html=True)

                original = np.array(Image.open(filename))
                col1, col2 = st.columns(2)
                col1.header("Оригинал")
                col1.image(original, width=350)
                
                to_img = pred * 255
                cv2.imwrite(user_dir + filename.split('/')[1], to_img)
                zip_obj.write(user_dir + filename.split('/')[1])
                
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
        
        # os.rmdir(user_dir)
        # os.rmdir(user_dir)


if __name__ == '__main__':
    main()
