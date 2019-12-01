"""
Testing trained models performance methods
"""
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import glob
from keras.preprocessing import image
from sklearn.metrics import precision_score, recall_score,f1_score, confusion_matrix


import os
import numpy as np

def predict_model(model,predict_generator,batch_size):
    """
    Predict classes for a test set and evaluate the model performance

    :param model: Trained model to test
    :param predict_generator: Directory generator for input data.
    :param batch_size: Batch size to load
    :return precision: Precision metric for test samples
    :return recall: Recall metric for test samples
    """

    nb_predict_samples = predict_generator.samples
    predict_classes = model.predict_generator(generator=predict_generator,steps=np.ceil(nb_predict_samples/batch_size),
                                       max_queue_size=1, workers=1,use_multiprocessing=False)


    predict_classes = np.argmax(predict_classes,axis=1)
    precision = precision_score(predict_generator.classes,predict_classes,average='macro')
    recall = recall_score(predict_generator.classes,predict_classes,average='macro')
    f1 = f1_score(predict_generator.classes,predict_classes,average='macro')

    for index, predict_clase in enumerate(predict_classes):
        if predict_clase != predict_generator.classes[index]:
            print(predict_generator.filenames[index])

    return precision, recall, f1


def get_confusion_matrix(model,predict_generator,batch_size):
    """
    Get the confusion matrix of a trained model

    :param model: Trained model to test
    :param predict_generator: Directory generator for input data. If nº inputs == 1, set predict_generator = predict_data
    :param batch_size: Batch size to load
    :return cf_matrix: Confusion matrix
    """

    nb_predict_samples = predict_generator.samples
    predict_classes = model.predict_generator(generator=predict_generator,steps=(nb_predict_samples//batch_size),
                                       max_queue_size=1)

    predict_classes = np.argmax(predict_classes,axis=1)

    return confusion_matrix(predict_generator.classes,predict_classes)