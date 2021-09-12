import streamlit as st
import gdown
from PIL import Image
import numpy as np
import custom.models
from config import Cfg
from production import get_predictions
from zipfile import ZipFile
import nibabel as nib
import random
import string
import os
import base64
import cv2

drive_link = 'https://drive.google.com/uc?id=1-tadxTBTRyru10rNNI0y4UcdntMK7hdh'


@st.cache
def download_model():
    gdown.cached_download(drive_link, quiet=False)


def get_random_string():
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(7)) + '/'


def read_files(files):
    folder_name = get_random_string()
    path = 'images/' + folder_name
    os.mkdir(path)
    imgs = list()
    for file in files:
        imgs.append([])
        if '.nii' in file.name:
            nii_path = path + file.name
            open(nii_path, 'wb').write(file.getvalue())
            images = nib.load(nii_path)
            images = np.array(images.dataobj)
            images = np.moveaxis(images, -1, 0)
            os.remove(nii_path)
            
            for idx, image in enumerate(images):
                image = window_image(image, -600, 1500)
                image += abs(np.min(image))
                image = image / np.max(image)
                image_path = path + file.name.split('.')[0] + '.png'
                cv2.imwrite(image_path, image * 255)
                
                imgs[-1].append(image_path)
                
        else:        
            with open(path + file.name, 'wb') as f:
                f.write(file.getvalue())

            imgs[-1].append(path + file.name)
    return imgs, folder_name


def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image


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
    filenames = st.file_uploader('Выберите или ператащите сюда снимки', type=['png', 'jpeg', 'jpg', '.nii', '.nii.gz'],
                                 accept_multiple_files=True)

    multi_class = st.checkbox(label='Мульти-классовая сегментация', value=True)

    if st.button('Загрузить') and filenames:
        print(filenames)
        images, folder_name = read_files(filenames)
        
        user_dir = "segmentations/" + folder_name
        os.mkdir(user_dir)
        
        cfg = Cfg(multi_class)
        zip_obj = ZipFile('segmentations.zip', 'w')
        with st.expander("Информация о каждом фото"):
            info = st.info('Делаем предсказания, пожалуйста, подождите')
            for image_list in images:
                for filename, pred in zip(image_list[:2], get_predictions(cfg, image_list[:2])):    
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
