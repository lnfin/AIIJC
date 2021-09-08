import streamlit as st
import gdown
import torch
from custom.models import DeepLabV3
from data_functions import Covid19Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from config import Cfg as cfg
from production import get_predictions
from utils import discretize_segmentation_maps

drive_link = 'https://drive.google.com/uc?id=1-tadxTBTRyru10rNNI0y4UcdntMK7hdh'  # example.pth


@st.cache
def download_model():
    gdown.cached_download(drive_link, "deeplabv3.pth", quiet=False)


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
    download_model()

    st.title('Сегментация поражения легких коронавирусной пневмонией')

    st.subheader("Загрузка файлов")
    filenames = st.file_uploader('Выберите или ператащите сюда снимки', type=['png', 'jpeg', 'jpg'],
                                 accept_multiple_files=True)

    st.markdown('<b>Выберите типы сегментации</b>', unsafe_allow_html=True)

    first_check = st.checkbox(label='Область поражения одним цветом', value=True)
    second_check = st.checkbox(label='Отдельно отображены разные виды', value=True)

    if not (first_check or second_check):
        st.error('Выберите одну из сегментаций')

    if st.button('Загрузить') and filenames:
        print(filenames)
        images = read_files(filenames)
        print(len(images))

        st.info('Делаем предсказания, пожалуйста, подождите')
        outputs = []
        for image in get_predictions(cfg, 'example.pth', images):
            print(image.shape)
            outputs.append(image[0].cpu().squeeze())

        with st.expander("Информация о каждом фото"):
            for filename, output in zip(images, outputs):
                st.markdown(f'<h3>{filename}</h3>', unsafe_allow_html=True)

                original = np.array(Image.open(filename))
                col1, col2 = st.columns(2)
                col1.header("Оригинал")
                col1.image(original, width=350)

                if first_check:
                    col2.header("Общая сегментация")
                    print(output.shape)
                    print(sum(output.numpy()))
                    col2.image(output.numpy(), width=350)

                    col1, col2, col3 = st.columns([1, 2, 1])

                if second_check:
                    col2.header('Потиповая сегментация')
                    # original[output.numpy()] = (0, 0, 255)
                    col2.image(original, width=350)
                    # col2.image(Image.open('images/multiseg.png'), width=350)

                st.markdown('<br />', unsafe_allow_html=True)

        with st.expander("Скачать сегментации"):
            pass


if __name__ == '__main__':
    main()
