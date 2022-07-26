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


def big_dataset_builder(
        train_paths, test_paths, output_path="./data/big_dataset", quality_threshold=0.7
):
    """
    go through all the paths and extract data above threshold to build big general dataset
    :param paths:
    :param quality_threshold:
    :param rotations:
    :return:
    """

    def extractor(paths_list, output_path):
        n_imgs = 0
        n_images_per_set_list = []
        for path in paths_list:
            print(f"Extracting images from dataset at {path} ...")
            n_images_per_set = 0
            img_files = [
                f for f in os.listdir(path) if
                os.path.isfile(os.path.join(path, f)) and f.split(".")[-1] == "npz"
            ]

            # iterate through the list
            for img in img_files:
                quality = int(img.split(".")[1]) / 1000
                if quality >= quality_threshold:
                    unnormalized_img = np.load(os.path.join(path, img))["arr_0"]
                    # normalized_img = unnormalized_img / np.max(unnormalized_img)
                    normalized_img = (unnormalized_img - np.min(unnormalized_img)) / \
                                     (np.max(unnormalized_img) - np.min(unnormalized_img))

                    np.savez(output_path + f"/{n_imgs}.npz", normalized_img)
                    n_imgs += 1
                    n_images_per_set += 1

                    img_rot90 = np.rot90(normalized_img)
                    np.savez(output_path + f"/{n_imgs}.npz", img_rot90)
                    n_imgs += 1
                    n_images_per_set += 1

                    img_rot180 = np.rot90(normalized_img, k=2)
                    np.savez(output_path + f"/{n_imgs}.npz", img_rot180)
                    n_imgs += 1
                    n_images_per_set += 1

                    img_rot270 = np.rot90(normalized_img, k=3)
                    np.savez(output_path + f"/{n_imgs}.npz", img_rot270)
                    n_imgs += 1
                    n_images_per_set += 1

            n_images_per_set_list.append(n_images_per_set)   # divide this by 4 to get number without data augmentation
        return n_images_per_set_list

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # else:
    #     shutil.rmtree(output_path)
    #     os.mkdir(output_path)

    train_save_path = output_path + f"/train_quality_{quality_threshold}"
    if not os.path.exists(train_save_path):
        os.mkdir(train_save_path)
    else:
        shutil.rmtree(train_save_path)
        os.mkdir(train_save_path)

    test_save_path = output_path + f"/test_quality_{quality_threshold}"
    if not os.path.exists(test_save_path):
        os.mkdir(test_save_path)
    else:
        shutil.rmtree(test_save_path)
        os.mkdir(test_save_path)

    n_imgs_per_set_train = extractor(train_paths, train_save_path)
    print(n_imgs_per_set_train)

    n_imgs_per_set_test = extractor(test_paths, test_save_path)
    print(n_imgs_per_set_test)




if __name__ == "__main__":
    # QUALITY_TH = 0.7
    # data_splitter_v2(path="./data/actin", quality_threshold=QUALITY_TH, individual_norm=True)

    from matplotlib import pyplot as plt

    # path = "./data/big_dataset/train_quality_0.7"
    # n_test_above_th = len(
    #     [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
    # )
    # for idx in range(n_test_above_th):
    #     img = np.load(path + f"/{idx}.npz")["arr_0"]
    #     print(np.min(img), np.max(img))
    #
    # exit()

    train_paths = [
        "./data/actin/train_all", "./data/camkii/CaMKII_Neuron/train", "./data/lifeact/LifeAct_Neuron/train",
        "./data/psd/PSD95_Neuron/train", "./data/tubulin/train"
    ]
    test_paths = [
        "./data/actin/test_all", "./data/camkii/CaMKII_Neuron/test", "./data/lifeact/LifeAct_Neuron/test",
        "./data/psd/PSD95_Neuron/test", "./data/tubulin/test"
    ]

    big_dataset_builder(train_paths, test_paths)

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
