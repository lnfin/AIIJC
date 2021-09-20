import streamlit as st
from PIL import Image
import numpy as np
import custom.models
from zipfile import ZipFile
import torch
import os
import base64
import cv2
from production import read_files, get_setup, make_masks, create_folder
import shutil


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
    models, transforms = get_setup()

    if st.button('Загрузить') and filenames:
        images, folder_name = read_files(filenames)

        if not images:
            st.error('Неправильный формат или название файла')
        else:
            user_dir = "segmentations/" + folder_name
            create_folder(user_dir)
            create_folder(os.path.join(user_dir, 'segmentations'))
            create_folder(os.path.join(user_dir, 'annotations'))

            zip_obj = ZipFile(user_dir + 'segmentations.zip', 'w')
            with st.expander("Информация о каждом фото"):
                info = st.info('Делаем предсказания, пожалуйста, подождите')
                for paths in images:
                    for img, annotation, path in make_masks(paths[:2], models, transforms, multi_class):
                        original_path = path
                        name = path.split('/')[-1].split('.')[0]
                        name = name.replace('\\', '/')

                        annotation_path = os.path.join(user_dir, 'annotations', name + '_annotation.txt')
                        with open(annotation_path, mode='w') as f:
                            f.write(annotation)
                        path = os.path.join(user_dir, 'segmentations', name + '_mask.png')

                        info.empty()
                        st.markdown(f'<h3>{name}</h3>', unsafe_allow_html=True)
                        st.markdown(annotation)

                        original = np.array(Image.open(original_path))
                        col1, col2 = st.columns(2)
                        col1.header("Оригинал")
                        col1.image(original, width=350)

                        cv2.imwrite(path, img)
                        zip_obj.write(path)
                        zip_obj.write(annotation_path)

                        col2.header("Сегментация")
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        col2.image(img / 255, width=350)

                        st.markdown('<br />', unsafe_allow_html=True)

                zip_obj.close()

            with st.expander("Скачать сегментации"):
                with open(os.path.join(user_dir, 'segmentations.zip'), 'rb') as file:
                    st.download_button(
                        label="Архив сегментаций",
                        data=file,
                        file_name="segmentations.zip",
                    )

                # shutil.rmtree(os.path.join('segmentations', folder_name))
                # shutil.rmtree(os.path.join('images', folder_name))


if __name__ == '__main__':
    main()
