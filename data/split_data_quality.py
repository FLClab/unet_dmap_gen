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


def data_splitter_v2(path="./data/actin", quality_threshold=0.7, individual_norm=False, rotations=True):
    """
    # TODO : implement individual img normalisation
    Splits the data :)
    :param path:
    :param quality_threshold:
    :param individual_norm:
    :return:
    """

    test_data_path = path + "/test_all"
    train_data_path = path + "/train_all"

    new_test_dir = path + f"/test_quality_{quality_threshold}"
    if not os.path.exists(new_test_dir):
        os.mkdir(new_test_dir)
    else:
        shutil.rmtree(new_test_dir)
        os.mkdir(new_test_dir)

    new_train_dir = path + f"/train_quality_{quality_threshold}"
    if not os.path.exists(new_train_dir):
        os.mkdir(new_train_dir)
    else:
        shutil.rmtree(new_train_dir)
        os.mkdir(new_train_dir)

    current_min = 999999999999
    current_max = 0
    print(f"Extracting test data with quality above {quality_threshold}")
    test_idx = 0
    for idx, name in enumerate(os.listdir(test_data_path)):
        if name.split(".")[-1] == "npz":
            if int(name.split("-")[-1].split(".")[1]) >= 1000 * quality_threshold:
                img = np.load(test_data_path + "/" + name)["arr_0"]
                unnormalized_img = np.floor(img * (2 ** 16 - 1) - 2 ** 15)

                img_min = np.min(unnormalized_img)
                if img_min < current_min:
                    current_min = img_min
                img_max = np.max(unnormalized_img)
                if img_max > current_max:
                    current_max = img_max

                if individual_norm:
                    normalized_img = unnormalized_img / np.max(unnormalized_img)
                    np.savez(new_test_dir + f"/{test_idx}.npz", normalized_img)
                else:
                    np.savez(new_test_dir + f"/{test_idx}.npz", unnormalized_img)

                test_idx += 1

    print(f"Extracting train data with quality above {quality_threshold}")
    train_idx = 0
    for idx, name in enumerate(os.listdir(train_data_path)):
        if name.split(".")[-1] == "npz":
            if int(name.split("-")[-1].split(".")[1]) >= 1000 * quality_threshold:
                img = np.load(train_data_path + "/" + name)["arr_0"]
                unnormalized_img = np.floor(img * (2 ** 16 - 1) - 2 ** 15)

                img_min = np.min(unnormalized_img)
                if img_min < current_min:
                    current_min = img_min
                img_max = np.max(unnormalized_img)
                if img_max > current_max:
                    current_max = img_max

                if individual_norm:
                    normalized_img = unnormalized_img / np.max(unnormalized_img)
                    np.savez(new_train_dir + f"/{train_idx}.npz", normalized_img)
                else:
                    np.savez(new_train_dir + f"/{train_idx}.npz", unnormalized_img)

                train_idx += 1

    if not individual_norm:
        print("Going through all images and normalizing them according to overall max")
        for idx, name in enumerate(os.listdir(new_test_dir)):
            img = np.load(os.path.join(new_test_dir, name))["arr_0"]
            normalized_img = img / current_max
            np.savez(new_test_dir + f"/{idx}.npz", normalized_img)

        for idx, name in enumerate(os.listdir(new_train_dir)):
            img = np.load(os.path.join(new_train_dir, name))["arr_0"]
            normalized_img = img / current_max
            np.savez(new_train_dir + f"/{idx}.npz", normalized_img)

    n_test_above_th = len(
        [name for name in os.listdir(new_test_dir) if os.path.isfile(os.path.join(new_test_dir, name))]
    )
    n_train_above_th = len(
        [name for name in os.listdir(new_train_dir) if os.path.isfile(os.path.join(new_train_dir, name))]
    )

    print(f"There are {n_train_above_th} train images with quality >= {quality_threshold} "
          f"({round(100 * n_train_above_th / (n_train_above_th + n_test_above_th), 4)} of images above threshold are train)")
    print(f"There are {n_test_above_th} test images with quality >= {quality_threshold} "
          f"({round(100 * n_test_above_th / (n_train_above_th + n_test_above_th), 4)} of images above threshold are test)")

    if rotations:
        biggest_idx = sorted(
            [int(name.split('.')[0]) for name in os.listdir(new_train_dir)
             if os.path.isfile(os.path.join(new_train_dir, name))]
        )[-1]

        augmented_idx = biggest_idx + 1
        for idx, name in enumerate(os.listdir(new_train_dir)):
            img = np.load(new_train_dir + "/" + name)["arr_0"]
            img_rot90 = np.rot90(img)
            np.savez(new_train_dir + f"/{augmented_idx}.npz", img_rot90)
            augmented_idx += 1
            img_rot180 = np.rot90(img, k=2)
            np.savez(new_train_dir + f"/{augmented_idx}.npz", img_rot180)
            augmented_idx += 1
            img_rot270 = np.rot90(img, k=3)
            np.savez(new_train_dir + f"/{augmented_idx}.npz", img_rot270)
            augmented_idx += 1

    n_train_augmented = len(
        [name for name in os.listdir(new_train_dir) if os.path.isfile(os.path.join(new_train_dir, name))]
    )

    print("Done")
    print(f"With rotations, there are {n_train_augmented} training images")



if __name__ == "__main__":
    QUALITY_TH = 0.7
    data_splitter_v2(path="./data/actin", quality_threshold=QUALITY_TH, individual_norm=True)

    # from matplotlib import pyplot as plt
    # print(":)")
    # # just to test, load an img from test_quality_0.7 and another from test_all and compare their values
    # # test_all img values should all be near 0.5
    # img_og_folder = np.load("./data/actin/test_all/0-0.757.npz")["arr_0"]
    # img_processed = np.load("./data/actin/train_quality_0.7/0.npz")["arr_0"]
    #
    # fig, axes = plt.subplots(1, 2)
    #
    # axes[0].imshow(img_og_folder)
    # axes[0].set_title(f"min = {np.min(img_og_folder)}, max = {np.max(img_og_folder)}")
    #
    # axes[1].imshow(img_processed)
    # axes[1].set_title(f"min = {np.min(img_processed)}, max = {np.max(img_processed)}")
    #
    # plt.show()
