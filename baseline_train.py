import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import cv2
from pandas.core.indexes import base
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow as tf
import tensorflow_addons as tfa
from glob import glob
import pickle
from sklearn.metrics import roc_curve
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# metachex materials
from metachex.configs.config import *
from metachex.dataloader import MetaChexDataset
from metachex.loss import Losses
from sklearn.metrics import roc_auc_score, precision_score, recall_score


def load_chexnet_pretrained(class_names=np.arange(14), weights_path='chexnet_weights.h5', 
                            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)):

    img_input = tf.keras.layers.Input(shape=input_shape)
    base_model = tf.keras.applications.densenet.DenseNet121(include_top=False, weights=None, 
                                                            input_tensor=img_input, pooling='avg')
    base_model.trainable = False


    x = base_model.output
    predictions = tf.keras.layers.Dense(len(class_names), activation="sigmoid", name="predictions")(x)
    model = tf.keras.models.Model(inputs=img_input, outputs=predictions)
    model.load_weights(weights_path)

    return model


def load_chexnet(output_dim):
    """
    output_dim: dimension of output
    """
    
    base_model_old = load_chexnet_pretrained()
    x = base_model_old.layers[-2].output ## remove old prediction layer
    
    ## The prediction head can be more complicated if you want
    predictions = tf.keras.layers.Dense(output_dim, name='prediction', activation='sigmoid')(x)
    chexnet = tf.keras.models.Model(inputs=base_model_old.inputs,outputs=predictions)
    return chexnet
    #base_model_old.trainable=False
    #return base_model_old


def mean_auroc(y_true, y_pred, eval=False):
    ## Note: roc_auc_score(y_true, y_pred, average='macro') #doesn't work for some reason -- didn't look into it too much
    aurocs = []
    with open("test_log.txt", "w") as f:
        for i in range(y_true.shape[1]):
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                aurocs.append(score)
            except ValueError:
                score = 0
            if eval:
                f.write(f"{dataset.unique_labels[i]}: {score}\n")
        mean_auroc = np.mean(aurocs)
        if eval:
            f.write("-----------------------\n")
            f.write(f"mean auroc: {mean_auroc}\n")
    if eval:
        print(f"mean auroc: {mean_auroc}")
    return mean_auroc


def mean_precision(y_true, y_pred):

    y_pred = np.where(y_pred >= 0.5, 1, 0)
    precisions = []
    for i in range(y_true.shape[1]):
        score = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        precisions.append(score)
    mean_precision = np.mean(precisions)
    
    return mean_precision


def mean_recall(y_true, y_pred):
    ## TODO: fix
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    recalls = []
    for i in range(y_true.shape[1]):
        score = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        recalls.append(score)
    mean_recall = np.mean(recalls)
    
    return mean_recall


def train():
    checkpoint_path = "training_progress/cp.ckpt" # path for saving model weights
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)
    
    epochs = 150
    hist = chexnet.fit(dataset.train_ds,
                validation_data=dataset.val_ds,
                epochs=epochs,
                steps_per_epoch=dataset.train_steps_per_epoch, ## size(train_ds) * 0.125 * 0.1
                validation_steps=dataset.val_steps_per_epoch, ## size(val_ds) * 0.125 * 0.2
                batch_size=dataset.batch_size ## 8
    # #             class_weight=class_weights,
    #             callbacks=[cp_callback]
                    )

    # with open('./trainHistoryDict', 'wb') as file_pi:
    #         pickle.dump(hist.history, file_pi)

    return hist


def compile():
    class_weights, indiv_class_weights, _ = dataset.get_class_weights(one_cap=True)

    loss_fn = Losses(class_weights, batch_size=dataset.batch_size)

    # output_dim = len(unique_labels)
    chexnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                    loss=loss_fn.weighted_binary_crossentropy(),
    #                   loss_weights=1e5,
    #                 loss='binary_crossentropy',
                    #metrics=[tf.keras.metrics.AUC(multi_label=True),  
                    metrics=[mean_auroc, #mean_precision, mean_recall, 'binary_accuracy', 'accuracy', 
                            tfa.metrics.F1Score(average='micro',num_classes=dataset.num_classes_multitask)], 
                    #        tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
                    run_eagerly=True)


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"]=""
    tf.test.is_gpu_available()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Instantiate dataset
    dataset = MetaChexDataset()
    
    # Grab training dataset
    train_ds = dataset.train_ds

    # Load CheXNet
    chexnet = load_chexnet(dataset.num_classes_multitask)
    chexnet.trainable = True
    
    print(chexnet.summary())
    
    # Compile
    compile()

    # Train
    hist = train()

    # Evaluate
    y_test_true = dataset.test_ds.get_y_true() 
    y_test_pred = chexnet.predict(dataset.test_ds, verbose=1)
    mean_auroc(y_test_true, y_test_pred, eval=True)


