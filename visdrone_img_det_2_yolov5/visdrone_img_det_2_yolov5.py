import pandas as pd
import cv2
import os

dir_list = ['VisDrone2019-DET-train', 'VisDrone2019-DET-test-dev', 'VisDrone2019-DET-val']

for dir in dir_list:
    print("----------------------------------")
    print(dir)
    print("----------------------------------")
    dir_locatin = '/media/2TB_1/Rahmani/downloads/datasets/visdrone/image_detection/' + dir
    txt_entries = os.listdir(dir_locatin + '/annotations')
    try:
        os.mkdir(dir_locatin + '/new_annotations')
    except:
        print("directory exist!")

    for txt_file in txt_entries:
        print(txt_file)
        im = cv2.imread(dir_locatin + '/images/' + txt_file[:-3] + 'jpg')

        im_h = im.shape[0]
        im_w = im.shape[1]

        df = pd.read_csv(dir_locatin + '/annotations/' + txt_file, header=None)
        data_list = []

        for index, row in df.iterrows():
            data_dict = {}
            data_dict['category'] = row[5]
            data_dict['x_center'] = "{:.6f}".format((row[0] + (row[2] / 2)) / im_w)
            data_dict['y_center'] = "{:.6f}".format((row[1] + (row[3] / 2)) / im_h)
            data_dict['width'] = "{:.6f}".format(row[2] / im_w)
            data_dict['height'] = "{:.6f}".format(row[3] / im_h)
            data_list.append(data_dict)

        new_df = pd.DataFrame(data_list)

        new_df.to_csv(dir_locatin + '/new_annotations/' + txt_file, header=None, index=None, sep=' ', mode='a')
