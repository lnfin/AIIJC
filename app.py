import streamlit as st
import numpy as np
import custom.models
from zipfile import ZipFile
import os
import cv2
from production import read_files, get_setup, create_folder
from inference import make_masks
import pandas as pd
from pydicom import dcmread


@st.cache(show_spinner=False, allow_output_mutation=True)
def cached_get_setup():
    return get_setup()


def main():
    st.set_page_config(page_title='Covid Segmentation')  # page_icon = favicon

    st.markdown(
        f"""
    <style>
        .sidebar .sidebar-content {{
            background: url("https://i.ibb.co/XSg54H1/image-2021-10-15-00-43-45.png");
            background-repeat: repeat;
            background-size: 100% 100%;
    }}
        .reportview-container {{
            background: url("https://i.ibb.co/XSg54H1/image-2021-10-15-00-43-45.png");
            background-repeat: repeat;
            background-size: 100% 100%;
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

    models, transforms = cached_get_setup()

    st.title('Сегментация поражения легких коронавирусной пневмонией')

    st.subheader("Загрузка файлов")
    filenames = st.file_uploader('Выберите или ператащите сюда снимки', type=['.png', '.dcm', '.rar', '.zip'],
                                 accept_multiple_files=True)

    multi_class = st.checkbox(label='Мульти-классовая сегментация', value=False)

    if st.button('Загрузить') and filenames:
        # Reading files
        info = st.info('Идет разархивация, пожалуйста, подождите')
        paths, folder_name = read_files(filenames)
        info.empty()
        if not paths or paths == [[]]:
            st.error('Неправильный формат или название файла')
        else:
            user_dir = "segmentations/" + folder_name

            # creating folders
            create_folder(user_dir)
            create_folder(os.path.join(user_dir, 'segmentations'))
            create_folder(os.path.join(user_dir, 'annotations'))

            binary_anno = '''
            <b>Бинарная сегментация:</b>\n
            <content style="color:Yellow">●</content> Всё повреждение\n
            '''

            multi_anno = '''
            <b>Мульти-классовая сегментация:</b>\n
            <content style="color:#00FF00">●</content> Матовое стекло\n
            <content style="color:Red">●</content> Консолидация\n
            '''

            all_zip = []
            all_stats = []
            for _paths in paths:
                stats = []
                mean_annotation = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float64)

                # Loading menu
                name = _paths[0].split('/')[-1].split('.')[0].replace('\\', '/')

                zip_obj = ZipFile(user_dir + f'segmentations_{name}.zip', 'w')
                all_zip.append(f'segmentations_{name}.zip')
                # Display file/patient name
                with st.expander(f"Информация о {name}"):
                    if multi_class:
                        st.markdown(multi_anno, unsafe_allow_html=True)
                    else:
                        st.markdown(binary_anno, unsafe_allow_html=True)

                    info = st.info(f'Делаем предсказания , пожалуйста, подождите')
                    for idx, data in enumerate(make_masks(_paths, models, transforms, multi_class)):
                        img, orig_img, img_to_dicom, annotation, path, _mean_annotation = data
                        info.empty()

                        # Вывод каждого второго    
                        if idx % 2 == 0:
                            st.subheader('Срез №' + str(idx + 1))

                            col1, col2 = st.columns(2)

                            col1.header("Оригинал")
                            col1.image(orig_img, width=350)

                            # show segmentation
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img_to_display = img / 255  # to [0;1] range
                            # print(img.shape, img.dtype, img)
                            col2.header("Сегментация")
                            col2.image(img_to_display, width=350)

                            if multi_class:
                                anno = f'''
                                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Левое</b>&nbsp;|&nbsp;<b>Правое</b>\n
                                <b>Матовое стекло:&nbsp;</b> {annotation['ground_glass'][0]:.2f}% | {annotation['ground_glass'][1]:.2f}%\n
                                <b>Консолидация:&nbsp;&nbsp;&nbsp;</b> {annotation['consolidation'][0]:.2f}% | {annotation['consolidation'][1]:.2f}%\n
                                    '''
                                col2.markdown(anno, unsafe_allow_html=True)

                        mean_annotation += _mean_annotation
                        # Store statistics
                        # print(img_to_dicom.shape)
                        # print(np.unique(img_to_dicom))

                        # if path.endswith('.dcm'):
                        img_to_save = img.astype(np.uint8)
                        if not path.endswith('.png') and not path.endswith('.jpg') and not path.endswith('.jpeg'):
                            ds = dcmread(path)
                            ds.PhotometricInterpretation = 'RGB'
                            ds.SamplesPerPixel = 3
                            ds.BitsAllocated = 8
                            ds.BitsStored = 8
                            ds.HighBit = 7
                            ds.add_new(0x00280006, 'US', 0)
                            ds.is_little_endian = True
                            ds.fix_meta_info()
                            # save pixel data and dicom file
                            ds.PixelData = img_to_save.tobytes()
                            path = path.replace('images', 'segmentations')
                            ds.save_as(path)
                            zip_obj.write(path)
                        # print(img_to_save.dtype, np.unique(img_to_save))

                        stat = {'id': idx + 1}
                        if multi_class:
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
                        else:
                            stat['left lung'] = annotation['disease'][0]
                            stat['right lung'] = annotation['disease'][1]
                            stat['both lung'] = stat['left lung'] + stat['right lung']

                        stats.append(stat)

                        info = st.info(f'Делаем предсказания , пожалуйста, подождите')
                    info.empty()
                    # Display statistics
                    df = pd.json_normalize(stats)
                    if multi_class:
                        df.columns = [
                            np.array(["ID", "left lung", "", "right lung", " ", "both", "  "]),
                            np.array(
                                ["", "Ground glass", "Consolidation", "Ground glass", "Consolidation", "Ground glass",
                                 "Consolidation"])
                        ]
                        df = df.append(pd.Series([
                            -1,
                            mean_annotation[0][2] / mean_annotation[0][0],
                            mean_annotation[0][1] / mean_annotation[0][0],
                            mean_annotation[1][2] / mean_annotation[1][0],
                            mean_annotation[1][1] / mean_annotation[1][0],
                            mean_annotation[0][2] / mean_annotation[0][0] + mean_annotation[1][2] / mean_annotation[1][
                                0],
                            mean_annotation[0][1] / mean_annotation[0][0] + mean_annotation[1][1] / mean_annotation[1][
                                0]
                        ], index=df.columns), ignore_index=True)

                        df['ID'] = df['ID'].astype('int32').replace(-1, '3D').astype('str')

                        df[["left lung", "", "right lung", " ", "both", "  "]] = df[
                            ["left lung", "", "right lung", " ", "both", "  "]].round(1).applymap('{:.1f}'.format)

                    else:
                        df.columns = np.array(["ID", "left lung", "right lung", "both"])

                        df['ID'] = df['ID'].astype('int32').replace(-1, '3D').astype('str')

                        df = df.append(pd.Series([
                            -1,
                            mean_annotation[0][1] / mean_annotation[0][0],
                            mean_annotation[1][1] / mean_annotation[1][0],
                            mean_annotation[0][1] / mean_annotation[0][0] + mean_annotation[1][1] / mean_annotation[1][
                                0]
                        ], index=df.columns), ignore_index=True)

                        df['ID'] = df['ID'].astype('int32').replace(-1, '3D').astype('str')

                        df[["left lung", "right lung", "both"]] = df[["left lung", "right lung", "both"]].round(
                            1).applymap('{:.1f}'.format)

                    st.dataframe(df)
                    df.to_excel(os.path.join(user_dir, f'statistics_{name}.xlsx'))

                    all_stats.append(f'statistics_{name}.xlsx')
                    zip_obj.close()

                # annotation_path = os.path.join(user_dir, 'annotation.txt')
                # with open(annotation_path, mode='w') as f:
                #         f.write(color_annotations)  
                # zip_obj.write(annotation_path)

            with st.expander("Скачать сегментации"):
                for zip_file in all_zip:
                    with open(os.path.join(user_dir, zip_file), 'rb') as file:
                        st.download_button(
                            label=zip_file,
                            data=file,
                            file_name=zip_file)

                for stat_file in all_stats:
                    with open(os.path.join(user_dir, stat_file), 'rb') as file:
                        st.download_button(
                            label=stat_file,
                            data=file,
                            file_name=stat_file
                        )


if __name__ == '__main__':
    main()
