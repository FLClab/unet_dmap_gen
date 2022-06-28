import numpy
import json
import datetime
import os

from tqdm import tqdm

# import dataset
#
#
# def save_json(d, file):
#     """
#     Saves the dict to the given file
#     :param d: A dictionary
#     :param file: A file where to save the data
#     """
#     json.dump(d, open(file, "w"), sort_keys=True, indent=4)
#
#
# def load_json(file):
#     """
#     Loads the dict from the json file
#     :param file: The file to open
#     :returns : A dict of the data
#     """
#     return json.load(open(file, "r"))
#
#
# def get_statistics(data):
#     """
#     Computes the statistics from the dataset
#     :param data: A `dict` of the training dataset
#     """
#     images_names = data["image"]
#     output = {}
#     for image_name in tqdm(images_names):
#         image, coords = dataset.load_image(image_name)
#         if isinstance(image, (type(None))):
#             continue
#         output[image_name] = {
#             "max": numpy.max(image, axis=(1, 2)).tolist(),
#             "min": numpy.min(image, axis=(1, 2)).tolist()
#         }
#     save_json(output, "./dataset/metadata_{}.json".format(datetime.date.today()))
#
#
# def print_params(trainer_params):
#     """
#     Prints trainer params
#     :params trainer_params: A `dict` of the training parameters
#     """
#
#     def print_center(string, width):
#         """
#         Prints a string at the center of terminal
#         """
#         for _ in range(2): print("-" * width)
#         print(" " * (width // 2 - len(string) // 2) + string)
#         for _ in range(2): print("-" * width)
#
#     def format_dict(d, current_string=""):
#         """
#         Formats a `dict` to print
#         """
#         strings = []
#         keys = sorted(list(d.keys()), key=lambda item: len(item))
#         for key, values in d.items():
#             prev_spaces = " " * (len(keys[-1]) - len(key) + len(current_string))
#             string = f"{prev_spaces}{key}    :    "
#             if isinstance(values, (dict)):
#                 string += "\n"
#                 values = format_dict(values, string)
#             strings.append("".join((string, str(values))))
#         return "\n".join(strings)
#
#     def print_strings(strings, width):
#         """
#         Prints formated params at center window
#         """
#         keys = sorted(list(strings.split("\n")), key=lambda item: len(item))
#         add_len = (width - len(keys[-1])) // 2
#         for string in strings.split("\n"):
#             print(" " * add_len + string)
#
#     try:
#         rows, columns = map(int, os.popen('stty size', 'r').read().split())
#     except ValueError:
#         rows, columns = 42, 100
#
#     print_center("Training parameters", columns)
#     strings = format_dict(trainer_params)
#     print_strings(strings, columns)
#     print_center("", columns)


class EarlyStopper:
    """
    Class that implements an early stopper.
    :param patience: The patience of the stopper
    :param threshold: The difference threshold to use for the stopper
    """

    def __init__(self, patience, threshold):
        self.patience = patience
        self.threshold = threshold
        self.training, self.validation = [], []

    def step(self, train_loss, valid_loss):
        """
        Method to add the training loss and the validation loss to the EarlyStopper
        class.
        :param train_loss: The training loss
        :param valid_loss: The validation loss
        :raises : An OSError if the difference for the past `patience` steps were
                  all above the preset threshold
        """
        self.training.append(train_loss)
        self.validation.append(valid_loss)

        if len(self.training) < self.patience:
            pass
        else:
            diff = numpy.subtract(self.validation[-self.patience:], self.training[-self.patience:])
            if all(diff > self.threshold):
                raise OSError(
                    "Early stopping. The difference for the past {} steps had all values higher than the threshold {}.".format(
                        self.patience, self.threshold))


if __name__ == "__main__":
    get_statistics(load_json("./dataset/training_2019-06-26.json"))