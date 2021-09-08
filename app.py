import streamlit as st
import gdown
import torch
from custom.models import DeepLabV3
from data_functions import Covid19Dataset
from torch.utils.data import DataLoader
from PIL import Image

drive_link = 'https://drive.google.com/uc?id=1vB0iLcF1OQBcihQV-OyCmFx1Rc8BwjcA'  # example.pth


@st.cache
def download_model():
    gdown.cached_download(drive_link, "example.pth", quiet=False)


@st.cache
def read_files(files):
    imgs = list()
    for file in files:
        with open('images/' + file.name, 'wb') as f:
            f.write(file.getvalue())

        imgs.append(file.name)
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
        print(images)
        print(len(images))
        with st.expander("Информация о каждом фото"):
            for image in images:
                st.markdown(f'<h3>{image}</h3>', unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                col1.header("Оригинал")
                col1.image(Image.open('images/original.jpeg'), width=350)

                if first_check:
                    col2.header("Общая сегментация")
                    col2.image(Image.open('images/multiseg.png'), width=350)

                    col1, col2, col3 = st.columns([1, 2, 1])

                if second_check:
                    col2.header('Потиповая сегментация')
                    col2.image(Image.open('images/multiseg.png'), width=350)

                st.markdown('<br />', unsafe_allow_html=True)

        with st.expander("Скачать сегментации"):
            pass


if __name__ == '__main__':
    main()
