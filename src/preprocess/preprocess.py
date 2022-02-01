# get libraries
import concurrent.futures
import pickle
import time
import cv2
import os
from matplotlib import pyplot as plt


# image understanding
def plot_hist_sample(sample, chn):
    img = sample

    if chn == 0:
        histr = cv2.calcHist([img], [chn], None, [256], [0, 256])
        plt.plot(histr)
        plt.xlim([0, 256])
        plt.show()

    else:
        histr = cv2.calcHist([img], [chn], None, [256], [0, 256])
        plt.plot(histr)
        plt.xlim([0, 256])
        plt.show()


# image statistical test
def stats_test(sample, label_name, cls_num, chn):
    pass


# preprocess, filter , check data distributions/ statistics
def preprocess(test=None):
    global down_size
    processed_dir = os.path.join(out_dir + process_name)
    os.makedirs(processed_dir, exist_ok=True)

    labels = os.listdir(raw_dir)
    if not labels:
        raise IOError('labels is empty')
    image_label = []

    for lbl in labels:

        fpath = os.path.join(raw_dir, lbl)
        cls_num = labels.index(lbl)
        for imgs in os.listdir(fpath):
            if imgs.endswith("pgm"):
                raw_img = cv2.imread(os.path.join(fpath, imgs), cv2.IMREAD_GRAYSCALE)
                down_size = cv2.resize(raw_img, (c, r), interpolation=cv2.INTER_LINEAR)
                image_label.append({'image': down_size, 'label': cls_num})

                # Display images
                # cv2.imshow('Image Down Size by defining height and width', down_size)
                # cv2.waitKey()
                # plot_hist_sample(down_size,chn)

    if test is None:
        # write features and labels to output file
        print(len(image_label))
        with open(os.path.join(processed_dir, 'image_label'), 'wb') as f:
            pickle.dump(image_label, f)

    else:
        with open(os.path.join(processed_dir, 'test'), 'wb') as f:
            pickle.dump(down_size, f)

    # sample output
    print(image_label[0]['image'])
    print(len(image_label))


if __name__ == '__main__':
    start = time.perf_counter()

    # paths and folder name variable
    in_dir = "GitHub/face_recognition/input"
    out_dir = "GitHub/face_recognition/output/"
    process_name = 'processed_YaleFaces_5x3r'
    raw_dir = os.path.join(in_dir, 'CroppedYale')
    # print(raw_dir)

    # variables
    c = 50  # no of columns
    r = 50  # no of rows
    chn = 0  # colour channel
    preprocess()

    end = time.perf_counter()
    print(f'processing finishes in {round(end - start, 2)} second(s)')
