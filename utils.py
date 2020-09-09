from __future__ import print_function
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import numpy as np
import os
import multiprocessing
workers = multiprocessing.cpu_count()//2
import tensorflow as tf

if tf.__version__[0] == "2":
    _IS_TF_2 = True
    import tensorflow.keras.backend as K
    from tensorflow.keras.utils import Sequence
    from tensorflow.keras.optimizers import Adam, SGD, RMSprop
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LambdaCallback
    from tensorflow.keras.layers import *
    from subpixel import *
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.callbacks import TensorBoard
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.python.client import device_lib
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.utils import to_categorical
else:
    _IS_TF_2 = False
    import keras
    import keras.backend as K
    from keras.utils.data_utils import Sequence
    from keras.optimizers import Adam, SGD, RMSprop
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LambdaCallback
    from keras.layers import *
    from subpixel import *
    from keras.models import Model, Sequential
    from keras.callbacks import TensorBoard
    from keras.preprocessing.image import ImageDataGenerator
    from tensorflow.python.client import device_lib
    from keras.regularizers import l2
    from keras.utils import to_categorical
    
from collections import Counter

from sklearn.utils import class_weight
import cv2
import glob
import random
from tqdm import tqdm
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
import itertools

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    trained_classes = classes
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=11)
    tick_marks = np.arange(len(classes))
    plt.xticks(np.arange(len(trained_classes)), classes, rotation=90,fontsize=9)
    plt.yticks(tick_marks, classes,fontsize=9)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, np.round(cm[i, j],2), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=7)
    plt.tight_layout()
    plt.ylabel('True label',fontsize=9)
    plt.xlabel('Predicted label',fontsize=9)
    return cm

# Fully connected CRF post processing function
def do_crf(im, mask, zero_unsure=True):
    colors, labels = np.unique(mask, return_inverse=True)
    image_size = mask.shape[:2]
    n_labels = len(set(labels.flat))
    d = dcrf.DenseCRF2D(image_size[1], image_size[0], n_labels)  # width, height, nlabels
    U = unary_from_labels(labels, n_labels, gt_prob=.7, zero_unsure=zero_unsure)
    d.setUnaryEnergy(U)
    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3,3), compat=3)
    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    # im is an image-array, e.g. im.dtype == np.uint8 and im.shape == (640,480,3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=im.astype('uint8'), compat=10)
    Q = d.inference(5) # 5 - num of iterations
    MAP = np.argmax(Q, axis=0).reshape(image_size)
    unique_map = np.unique(MAP)
    for u in unique_map: # get original labels back
        np.putmask(MAP, MAP == u, colors[u])
    return MAP
    # MAP = do_crf(frame, labels.astype('int32'), zero_unsure=False)
    
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
    

def get_VOC2012_classes():
    PASCAL_VOC_classes = {
        0: 'background', 
        1: 'airplane',
        2: 'bicycle',
        3: 'bird', 
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'table',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'potted_plant',
        17: 'sheep',
        18: 'sofa',
        19 : 'train',
        20 : 'tv',
        21 : 'void'
    }
    return PASCAL_VOC_classes


def Jaccard(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    pred_pixels = K.argmax(y_pred, axis=-1)
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        true_labels = K.equal(y_true[:,:,0], i)
        pred_labels = K.equal(pred_pixels, i)
        inter = tf.to_int32(true_labels & pred_labels)
        union = tf.to_int32(true_labels | pred_labels)
        legal_batches = K.sum(tf.to_int32(true_labels), axis=1)>0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        if _IS_TF_2:
            iou.append(K.mean(ious[legal_batches]))
        else:
            iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.math.is_nan(iou) if _IS_TF_2 else ~tf.debugging.is_nan(iou)
    iou = iou[legal_labels] if _IS_TF_2 else tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)

        