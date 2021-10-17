style = f"""
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
    """

legend_binary = '''
            <b>Бинарная сегментация:</b>\n
            <content style="color:Yellow">●</content> Всё повреждение\n
            '''

legend_multi = '''
            <b>Мульти-классовая сегментация:</b>\n
            <content style="color:#00FF00">●</content> Матовое стекло\n
            <content style="color:Red">●</content> Консолидация\n
            '''


def pretty_annotation(annotation):
    if 'ground_glass' in annotation.keys():
        annotation = f'''
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Левое</b>&nbsp;|&nbsp;<b>Правое</b>\n
            <b>Матовое стекло:&nbsp;</b> {annotation['ground_glass'][0]:.2f}% | {annotation['ground_glass'][1]:.2f}%\n
            <b>Консолидация:&nbsp;&nbsp;&nbsp;</b> {annotation['consolidation'][0]:.2f}% | {annotation['consolidation'][1]:.2f}%\n
                '''
    else:
        annotation = f'''
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Левое</b>&nbsp;|&nbsp;<b>Правое</b>\n
            <b>Повреждение:&nbsp;</b> {annotation['disease'][0]:.2f}% | {annotation['disease'][1]:.2f}%\n'''
    return annotation
