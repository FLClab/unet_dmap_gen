import numpy
import json
import datetime
import os

from tqdm import tqdm

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
