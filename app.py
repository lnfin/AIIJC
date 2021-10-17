import streamlit as st
import numpy as np
import custom.models
from zipfile import ZipFile
import os
from production import *
from inference import make_masks
import string_to_web as stw


@st.cache(show_spinner=False, allow_output_mutation=True)
def cached_get_setup():
    return get_setup()


def main():
    st.set_page_config(page_title='Covid Segmentation')  # page_icon = favicon

    st.markdown(
        stw.style,
        unsafe_allow_html=True)
    for folder in ['segmentations/', 'images/', 'checkpoints/']:
        create_folder(folder)

    models, transforms = cached_get_setup()

    st.title('Сегментация поражения легких коронавирусной пневмонией')

    st.subheader("Загрузка файлов")
    filepaths = st.file_uploader('Выберите или ператащите сюда снимки', type=['.png', '.dcm', '.rar', '.zip'],
                                 accept_multiple_files=True)

    multi_class = st.checkbox(label='Мульти-классовая сегментация', value=False)

    if st.button('Загрузить') and filepaths:
        # Reading files
        info = st.info('Идет разархивация, пожалуйста, подождите')
        user_folder = generate_folder_name()
        paths = read_web_files(filepaths, user_folder)
        info.empty()
        if not any(paths):
            st.error('Неправильный формат или название файла')
        else:
            user_dir = os.path.join('segmentations', user_folder)

            # creating folders
            create_folder(user_dir)
            create_folder(os.path.join(user_dir, 'segmentations'))
            create_folder(os.path.join(user_dir, 'annotations'))

            all_zip = []
            all_stats = []
            for idx, _paths in enumerate(paths):
                _paths.sort()
                stats = []
                mean_data = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float64)

                step = max(0.02 * len(_paths), 1)
                # Loading menu
                name = name_from_filepath(filepaths[idx].name)

                nifti = NiftiSaver()

                zip_obj = ZipFile(os.path.join(user_dir, f'{name}.zip'), 'w')
                all_zip.append(f'{name}.zip')
                # Display file/patient name
                with st.expander(f"Информация о {name}"):
                    # Legend for segmentation
                    st.markdown(stw.legend_multi if multi_class else stw.legend_binary, unsafe_allow_html=True)

                    info = st.info(f'Делаем предсказания , пожалуйста, подождите')
                    kol = 0
                    for idx, data in enumerate(make_masks(_paths, models, transforms, multi_class)):
                        img, orig_img, img_to_dicom, annotation, path, _mean_data = data
                        info.empty()

                        nifti.add(img_to_dicom)

                        mean_data += _mean_data
                        img_to_save = img.astype(np.uint8)
                        if not path.endswith('.png') and not path.endswith('.jpg') and not path.endswith('.jpeg'):
                            save_dicom(path, img_to_save)
                            zip_obj.write(path)

                        # annotation to statistic
                        stat = get_statistic(idx, annotation)
                        stats.append(stat)

                        # Only half of slices
                        if round(kol * step) == idx or int(kol * step) + 1 == idx:
                            st.subheader('Срез №' + str(idx + 1))

                            # Original image display
                            col1, col2 = st.columns(2)
                            col1.header("Оригинал")
                            col1.image(orig_img, width=350)

                            # Segmentation image display
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img_to_display = img / 255  # to [0;1] range
                            col2.header("Сегментация")
                            col2.image(img_to_display, width=350)

                            # Annotation display
                            pretty_annotation = stw.pretty_annotation(annotation)
                            col2.markdown(pretty_annotation, unsafe_allow_html=True)

                            kol += 1

                        info = st.info(f'Делаем предсказания , пожалуйста, подождите')
                    info.empty()
                    # Creating dataframe to display and save
                    df = create_dataframe(stats, mean_data)
                    # Display statistics
                    st.dataframe(df)
                    # Save statistics
                    df.to_excel(os.path.join(user_dir, f'{name}.xlsx'))
                    all_stats.append(f'{name}.xlsx')
                    print('PATH ', user_dir)
                    nifti.add(os.path.join(user_dir, f'{name}.nii'))
                    # Close zip
                    zip_obj.close()

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
