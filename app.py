import streamlit as st
from PIL import Image
import numpy as np
import custom.models
from zipfile import ZipFile
import os
import cv2
from production import read_files, get_setup, make_masks, create_folder, make_legend, window_image
import shutil
import pandas as pd
from pydicom import dcmread


@st.cache(show_spinner=False, allow_output_mutation=True)
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
    for folder in ['segmentations/', 'images/', 'checkpoints/']:
        create_folder(folder)

    st.title('Сегментация поражения легких коронавирусной пневмонией')

    st.subheader("Загрузка файлов")
    filenames = st.file_uploader('Выберите или ператащите сюда снимки', type=['.png', '.dcm', '.rar'],
                                 accept_multiple_files=True)

    multi_class = st.checkbox(label='Мульти-классовая сегментация', value=False)

    if st.button('Загрузить') and filenames:
        # Reading files
        info = st.info('Идет разархивация, пожалуйста, подождите')
        paths, folder_name = read_files(filenames)
        info.empty()

        print(paths)
        if not paths or paths == [[]]:
            st.error('Неправильный формат или название файла')
        else:
            user_dir = "segmentations/" + folder_name

            # creating folders
            create_folder(user_dir)
            create_folder(os.path.join(user_dir, 'segmentations'))
            create_folder(os.path.join(user_dir, 'annotations'))

            zip_obj = ZipFile(user_dir + 'segmentations.zip', 'w')

            gallery = []
            with st.expander("Статистика о пациенте"):
                info = st.info('Делаем предсказания, пожалуйста, подождите')
                for _paths in paths:
                    gallery.append([])
                    stats = []
                    data = np.array([[0, 0, 0], [0, 0, 0]])
                    for idx, (img, annotation, original_path, _data) in enumerate(
                            make_masks(_paths, models, transforms, multi_class)):
                        info.empty()
                        data += _data
                        # Display file/patient name
                        if idx == len(_paths) - 1:
                            name = _paths[0].split('/')[-1].split('.')[0].replace('\\', '/')[:-2]
                            st.markdown(f'<h3>{name}</h3>', unsafe_allow_html=True)

                        # Store statistics
                        stat = {}
                        if multi_class:
                            stat['id'] = idx + 1
                            stat['left lung'] = {
                                'Ground glass': annotation['ground_glass'][0],
                                'Consolidation': annotation['consolidation'][0]
                            }
                            stat['right lung'] = {
                                'Ground glass': annotation['ground_glass'][1],
                                'Consolidation': annotation['consolidation'][1]
                            }
                            stat['both lungs'] = {
                                'Ground glass': sum(annotation['ground_glass']),
                                'Consolidation': sum(annotation['consolidation'])
                            }
                            stats.append(stat)

                        # Store data to gallery
                        gallery[-1].append((original_path, img, annotation))
                    print(stats)

                    # Display statistics
                    df = pd.json_normalize(stats)
                    df.columns = [
                        np.array(["ID", "left lung", "", "right lung", " ", "both", "  "]),
                        np.array(["", "Ground glass", "Consolidation", "Ground glass", "Consolidation", "Ground glass",
                                  "Consolidation"])
                    ]

                    df = df.append(pd.Series([
                        -1,
                        df['left lung']['Ground glass'].mean(),

                        df['']['Consolidation'].mean(),
                        df['right lung']['Ground glass'].mean(),
                        df[' ']['Consolidation'].mean(),
                        df['both']['Ground glass'].mean(),
                        df['  ']['Consolidation'].mean()
                    ], index=df.columns), ignore_index=True)

                    df['ID'] = df['ID'].astype('int32').replace(-1, 'avg').astype('str')

                    df[["left lung", "", "right lung", " ", "both", "  "]] = df[
                        ["left lung", "", "right lung", " ", "both", "  "]].round(1).applymap('{:.1f}'.format)
                    st.dataframe(df)
                    df.to_excel(os.path.join(user_dir, 'statistics.xlsx'))
                    print(data)
                    if multi_class:
                        results = {
                            'cs_left': data[0][1] / data[0][0],
                            'gg_left': data[0][2] / data[0][0],
                            'cs_right': data[1][1] / data[1][0],
                            'gg_right': data[1][2] / data[1][0]
                        }
                    else:
                        results = {
                            'disease_left': data[0][1] / data[0][0],
                            'disease_right': data[1][1] / data[1][0]
                        }

                # annotation_path = os.path.join(user_dir, 'annotation.txt')
                # with open(annotation_path, mode='w') as f:
                #         f.write(color_annotations)  
                # zip_obj.write(annotation_path)

            color_annotations = '''
            <b>Binary mode:</b>\n
            <content style="color:Yellow">●</content> Всё повреждение\n
            
            <b>Multi mode:</b>\n
            <content style="color:#00FF00">●</content> Матовое стекло\n
            <content style="color:Red">●</content> Консолидация\n
            '''

            with st.expander("Галерея"):
                st.markdown(color_annotations, unsafe_allow_html=True)

                # for line in list(annotation.keys()):
                #     st.markdown(line)
                for patient in gallery:
                    for idx in range(0, len(patient), 2):
                        original_path, img, annotation = patient[idx]
                        st.subheader('Slice №' + str(idx + 1))

                        col1, col2 = st.columns(2)
                        # original image
                        original = dcmread(original_path).pixel_array
                        original = window_image(original)

                        col1.header("Оригинал")
                        col1.image(original, width=350)

                        # show segmentation
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = img / 255  # to [0;1] range
                        # print(img.shape, img.dtype, img)
                        col2.header("Сегментация")
                        col2.image(img, width=350)
                        if multi_class:
                            anno = f'''
                                             <b>Left</b>             |             <b>Right</b>\n
                            <b>Ground Glass:</b> {annotation['ground_glass'][0]:.2f}% | {annotation['ground_glass'][1]:.2f}%\n
                            <b>Consolidation:</b> {annotation['consolidation'][0]:.2f}% | {annotation['consolidation'][1]:.2f}%\n
                                '''
                            col2.markdown(anno, unsafe_allow_html=True)
            # download segmentation zip
            zip_obj.close()

            with st.expander("Скачать сегментации"):
                with open(os.path.join(user_dir, 'segmentations.zip'), 'rb') as file:
                    st.download_button(
                        label="Архив сегментаций",
                        data=file,
                        file_name="segmentations.zip")

                with open(os.path.join(user_dir, 'statistics.xlsx'), 'rb') as file:
                    st.download_button(
                        label="Статистика",
                        data=file,
                        file_name="statistcs.xlsx"
                    )


if __name__ == '__main__':
    main()
