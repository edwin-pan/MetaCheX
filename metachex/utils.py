import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score
from metachex.configs.config import *

colormap =  lambda x, N: np.array(matplotlib.cm.get_cmap('viridis')(x/N))


def load_chexnet_pretrained(class_names=np.arange(14), weights_path='chexnet_weights.h5', 
                            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)):

    img_input = tf.keras.layers.Input(shape=input_shape)
    base_model = tf.keras.applications.densenet.DenseNet121(include_top=False, weights=None,  ## uncomment
                                                            input_tensor=img_input, pooling='avg')
    base_model.trainable = False


    x = base_model.output
    predictions = tf.keras.layers.Dense(len(class_names), activation="sigmoid", name="predictions")(x)
    model = tf.keras.models.Model(inputs=img_input, outputs=predictions)
    model.load_weights(weights_path) ## uncomment

    return model


def load_chexnet(output_dim, embedding_dim=128):
    """
    output_dim: dimension of output
    """
    
    base_model_old = load_chexnet_pretrained()
    x = base_model_old.layers[-2].output ## remove old prediction layer
    
    ## The prediction head can be more complicated if you want
    embeddings = tf.keras.layers.Dense(embedding_dim, name='embedding', activation='relu')(x)
    normalized_embeddings = tf.nn.l2_normalize(embeddings, axis=-1)
    predictions = tf.keras.layers.Dense(output_dim, name='prediction', activation='sigmoid')(normalized_embeddings) # BASELINE: directly predict
    chexnet = tf.keras.models.Model(inputs=base_model_old.inputs,outputs=predictions)
    return chexnet
    #base_model_old.trainable=False
    #return base_model_old

def get_embedding_model(model):
    x = model.layers[-2].output
    chexnet_embedder = tf.keras.models.Model(inputs = model.input, outputs = x)
    return chexnet_embedder

def mean_auroc_baseline(y_true, y_pred):
    ## Note: roc_auc_score(y_true, y_pred, average='macro') #doesn't work for some reason -- didn't look into it too much
    aurocs = []
    with open("test_log.txt", "w") as f:
        for i in range(y_true.shape[1]):
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                aurocs.append(score)
            except ValueError:
                score = 0
        mean_auroc = np.mean(aurocs)
        if eval:
            f.write("-----------------------\n")
            f.write(f"mean auroc: {mean_auroc}\n")

    return mean_auroc

def mean_auroc(y_true, y_pred, dataset, eval=False):
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


def average_precision(y_true, y_pred, dataset):
    test_ap_log_path = os.path.join(".", "average_prec.txt")
    with open(test_ap_log_path, "w") as f:
        aps = []
        for i in range(y_true.shape[1]):
            ap = average_precision_score(y_true[:, i], y_pred[:, i])
            aps.append(ap)
            f.write(f"{dataset.unique_labels[i]}: {ap}\n")
        mean_ap = np.mean(aps)
        f.write("-------------------------\n")
        f.write(f"mean average precision: {mean_ap}\n")

        
def process_tSNE(features, learning_rate=10, perplexity=20):
    """ Computes tNSE embedding as array"""
    tsne = manifold.TSNE(n_components=2, init="random", learning_rate=learning_rate, random_state=0, perplexity=perplexity)
    encoded = tsne.fit_transform(features)
    return encoded

def plot_tsne(tsne_features, tsne_labels_one_hot, 
                              lables_names=None, 
                              num_subsample=None, 
                              visualize_class_list=None, 
                              plot_title='test', 
                              save_path='test.png'):

    tsne_labels = np.argmax(tsne_labels_one_hot, axis=-1)
    num_classes = tsne_labels_one_hot.shape[-1]
    num_samples = tsne_labels_one_hot.shape[0]

    if num_subsample is not None:
        # TODO:Visualize subset of samples
        pass

    if visualize_class_list is not None:
        # TODO: Visualize subset of classes
        pass

    if lables_names is None:
        plt.figure()
        plt.scatter(tsne_features[...,0], tsne_features[...,1], c=tsne_labels, cmap=plt.cm.get_cmap('viridis', num_classes))
        plt.title(plot_title)
        plt.colorbar(ticks=np.arange(num_classes))
        plt.tight_layout()
        plt.savefig(save_path)
    else:
        plt.figure()
        plt.title(plot_title)
        ax = plt.subplot(111)
        for i in range(num_classes):            
            scat = ax.scatter(
                tsne_features[tsne_labels==i,0], tsne_features[tsne_labels==i,1], 
                c=np.repeat(colormap(i, num_classes).reshape(-1,1), [tsne_features[tsne_labels==i,0].shape[0]]).reshape(4,-1).T,
                alpha=0.9,
                label=lables_names[i])

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                fancybox=True, shadow=True, ncol=5)

        plt.savefig(save_path)
    return

