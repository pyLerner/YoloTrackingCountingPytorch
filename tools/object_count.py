"""
    Подсчет количества значений классов объектов, записанных
    в runs/track/exp*/mot/*.txt
"""

import pandas as pd
from collections import Counter
# from pathlib import Path

"""
Классы из  COCO DateSet
    0	person
    1	bicycle
    2	car
    3	motorcycle
    4	airplane
    5	bus
    6	train
    7	truck
    8	boat
"""

class_names = [
    'Пешеходы',
    'Велосипеды',
    'Автомобили',
    'Мотоцикл',
    'Самолеты',
    'Автобусы',
    'Поезда',
    'Грузовики',
    'Лодки'
]


def track_object_count(mot_file: str) -> object:

    df = pd.read_csv(mot_file, sep=' ')

    df.columns = ['frame', 'objectID', 'X', 'Y', 'H', 'W',
                  'Question', 'Cls', 'Coord']

    print(df)

    # Массив уникальных objectID
    # print(df.objectID.unique())

    # cls_count = pd.DataFrame()    Для обработки в pandas без помощи Counter
    cls_list = []

    for object_id in df.objectID.unique():
        df_id = df[df.objectID == object_id][['objectID', 'Cls']]

        # Tuple[objectID, max_count_class] - чаще встречающийся класс для каждого objectId
        record = df_id.value_counts().idxmax()
        # print(record)
        # cls_count = pd.concat([cls_count, pd.DataFrame()])
        cls_list.append(record[1])

    counts = dict(Counter(cls_list))
    # print(counts)

    out_dict = {}
    for key, value in counts.items():
        new_key = class_names[key]
        out_dict[new_key] = counts[key]

    out_df = pd.DataFrame.from_dict(out_dict, orient='index', columns=['Кол-во'])
    print(out_df)
    print('Всего проехало через перекресток: ', end='')
    print(out_df['Кол-во'].sum())

    return out_df


if __name__ == "__main__":

    mot_file_path = (
        '/Users/pylerner/pythonProject/'
        'YoloTrackingPytorch/runs/track/'
        'exp11/mot/video/2-648-10min.webm.txt'
    )

    track_object_count(mot_file_path)
