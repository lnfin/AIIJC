import streamlit as st
from PIL import Image
import numpy as np
import custom.models
from zipfile import ZipFile
import os
import cv2
from production import read_files, get_setup, make_masks, create_folder, make_legend
import shutil


@st.cache
def cached_get_setup():
    return get_setup()


def main():
    models, transforms = cached_get_setup()
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
    for folder in ['segmentations/', 'images/']:
        create_folder(folder)

    st.title('Сегментация поражения легких коронавирусной пневмонией')

    st.subheader("Загрузка файлов")
    filenames = st.file_uploader('Выберите или ператащите сюда снимки', type=['png', 'jpeg', 'jpg', '.nii', '.nii.gz'],
                                 accept_multiple_files=True)

    multi_class = st.checkbox(label='Мульти-классовая сегментация', value=False)
    show_legend = st.checkbox(label='Легенда на картинке', value=False)

    if st.button('Загрузить') and filenames:
        paths, folder_name = read_files(filenames)
        if not paths:
            st.error('Неправильный формат или название файла')
        else:
            user_dir = "segmentations/" + folder_name

            # creating folders
            create_folder(user_dir)
            create_folder(os.path.join(user_dir, 'segmentations'))
            create_folder(os.path.join(user_dir, 'annotations'))

            zip_obj = ZipFile(user_dir + 'segmentations.zip', 'w')
            with st.expander("Информация о каждом фото"):
                info = st.info('Делаем предсказания, пожалуйста, подождите')
                for _paths in paths:
                    for img, annotation, original_path in make_masks(_paths, models, transforms, multi_class):
                        name = original_path.split('/')[-1].split('.')[0]
                        name = name.replace('\\', '/')

                        # saving annotation
                        annotation_path = os.path.join(user_dir, 'annotations', name + '_annotation.txt')
                        with open(annotation_path, mode='w') as f:
                            f.write(annotation)

                        info.empty()

                        # name and annotation
                        st.markdown(f'<h3>{name}</h3>', unsafe_allow_html=True)
                        if not show_legend:
                            for line in annotation.split('\n'):
                                st.markdown(line)

                        col1, col2 = st.columns(2)

                        # original image
                        original = np.array(Image.open(original_path))
                        col1.header("Оригинал")
                        col1.image(original, width=350)

                        # refactoring image
                        if show_legend:
                            img = make_legend(img, annotation)

                        # saving image
                        path = os.path.join(user_dir, 'segmentations', name + '_mask.png')
                        cv2.imwrite(path, img)

                        # adding in zip
                        zip_obj.write(path)
                        zip_obj.write(annotation_path)

                        # show segmentation
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = img / 255  # to [0;1] range
                        # print(img.shape, img.dtype, img)
                        col2.header("Сегментация")
                        col2.image(img, width=350)

                        st.markdown('<br />', unsafe_allow_html=True)

                zip_obj.close()

            # download segmentation zip
            with st.expander("Скачать сегментации"):
                with open(os.path.join(user_dir, 'segmentations.zip'), 'rb') as file:
                    st.download_button(
                        label="Архив сегментаций",
                        data=file,
                        file_name="segmentations.zip")


if __name__ == '__main__':
    main()
