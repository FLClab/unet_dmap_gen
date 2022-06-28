import os
import shutil
import numpy as np


# copy data from target folder into new folder, keeping only data which's quality is >= a threshold
def data_splitter(path="./data/actin", quality_threshold=0.7):
    # PATH = "./data/actin"
    # QUALITY_THRESHOLD = 0.7

    test_data_path = path + "/test_all"
    train_data_path = path + "/train_all"

    new_test_dir = path + f"/test_quality_{quality_threshold}"
    if not os.path.exists(new_test_dir):
        os.mkdir(new_test_dir)

    print(f"Extracting test data with quality above {quality_threshold}")
    for name in os.listdir(test_data_path):
        if name.split(".")[-1] == "npz":
            if int(name.split("-")[-1].split(".")[1]) >= 1000 * quality_threshold:
                shutil.copyfile(test_data_path + "/" + name, new_test_dir + "/" + name)

    n_test_above_th = len([name for name in os.listdir(new_test_dir) if os.path.isfile(os.path.join(new_test_dir, name))])

    new_train_dir = path + f"/train_quality_{quality_threshold}"
    if not os.path.exists(new_train_dir):
        os.mkdir(new_train_dir)

    print(f"Extracting train data with quality above {quality_threshold}")
    for name in os.listdir(train_data_path):
        if name.split(".")[-1] == "npz":
            if int(name.split("-")[-1].split(".")[1]) >= 1000 * quality_threshold:
                shutil.copyfile(train_data_path + "/" + name, new_train_dir + "/" + name)
    n_train_above_th = len([name for name in os.listdir(new_train_dir) if os.path.isfile(os.path.join(new_train_dir, name))])

    print("Done")
    print(f"There are {n_train_above_th} train images with quality >= {quality_threshold} "
          f"({round(100 * n_train_above_th / (n_train_above_th + n_test_above_th), 4)} of images above threshold are train)")
    print(f"There are {n_test_above_th} test images with quality >= {quality_threshold} "
          f"({round(100 * n_test_above_th / (n_train_above_th + n_test_above_th), 4)} of images above threshold are test)")


def data_splitter_v2(path="./data/actin", quality_threshold=0.7):
    """
    Splits the data
    :param path:
    :param quality_threshold:
    :return:
    """


if __name__ == "__main__":
    QUALITY_TH = 0.7
    data_splitter(path="./data/actin", quality_threshold=QUALITY_TH)
