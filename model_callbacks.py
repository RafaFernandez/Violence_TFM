"""
Auxiliar metrics used on CNN  performance's evaluation
"""

from keras.callbacks import Callback
from Violence.model_test import predict_model, get_confusion_matrix
import numpy as np



class metric_callback(Callback):
    """
    Callback on trainig stage that returns the metric value at the end of every epoch
    """

    def __init__(self, validation_generator=None, filename='', filename_matrix=''):
        """
        Create the variable associated to the class

        :param validation_generator: Keras Data generator
        :param filename: Filename for saving the model
        :param filename_matrix: Filename for saving the confusion matrix
        """
        self.validation_generator = validation_generator
        self.filename = filename
        self.filename_matrix = filename_matrix
        self.name_matrix = ""

        return

    def on_train_begin(self,logs=None):
        """
        Initialize the F1 values array at the beginning of the train.
        """

        self.f1s = []

        return



    def on_epoch_end(self, epoch, logs):
        """
        Get the Precision and Recall value and store it at the end of each training epoch
        :param logs: Output log with performance metrics
        """

        precision,recall,f1 = predict_model(self.model, predict_generator=self.validation_generator,
                                                batch_size=self.validation_generator.batch_size)


        logs['precision'] = precision
        logs['recall'] = recall
        logs['f1'] = f1
        self.f1s.append(f1)

        if f1 >= max(self.f1s):
            self.cf = get_confusion_matrix(self.model, self.validation_generator, self.validation_generator,
                                           self.validation_generator.batch_size)
            self.name_matrix = self.filename_matrix
            np.savetxt(self.name_matrix, self.cf, fmt='%d')
            self.model.save(self.filename)

        return

