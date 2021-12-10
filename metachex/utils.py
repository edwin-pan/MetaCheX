from typing import ByteString
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
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, f1_score

from metachex.configs.config import *
from sklearn.metrics.pairwise import euclidean_distances
from metachex.image_sequence import ImageSequence
import matplotlib.pyplot as plt

colormap =  lambda x, N: np.array(matplotlib.cm.get_cmap('viridis')(x/N))

def baby_conv(input_obj):
    model = tf.keras.models.Sequential()
    model.add(input_obj)
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(244, 244, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    return model

def load_chexnet(output_dim, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), multiclass=False, embedding_dim=128, baseline=False):
    """
    output_dim: dimension of output
    multiclass: whether or not prediction task is multiclass (vs. binary multitask)
    Note: multiclass argument is only relevant for baseline models
    """
    
    img_input = tf.keras.layers.Input(shape=input_shape)
    if baseline:
        base_model = tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet',  # use imagenet weights
                                                            input_tensor=img_input, pooling='avg')
    else:
        base_model = baby_conv(img_input)

    x = base_model.output
    
    ## The prediction head can be more complicated if you want
    embeddings = tf.keras.layers.Dense(embedding_dim, name='embedding', activation='relu')(x)
    normalized_embeddings = embeddings # tf.nn.l2_normalize(embeddings, axis=-1)
    if multiclass:
        activation = 'softmax'
    else:
        activation = 'sigmoid'
        
    # BASELINE: directly predict
    predictions = tf.keras.layers.Dense(output_dim, name='prediction', activation=activation)(normalized_embeddings)
    
    chexnet = tf.keras.models.Model(inputs=base_model.inputs,outputs=predictions)
    return chexnet

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

def mean_auroc(y_true, y_pred, dataset=None, eval=False, dir_path='.'):
    ## Note: roc_auc_score(y_true, y_pred, average='macro') #doesn't work for some reason -- didn't look into it too much
#     print(f'y_true: {y_true}')
#     print(f'y_pred: {y_pred}')
    aurocs = []
    test_auroc_log_path = os.path.join(dir_path, "auroc.txt")
    with open(test_auroc_log_path, "w") as f:
        for i in range(y_true.shape[1]):
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                aurocs.append(score)
            except ValueError:
#                 import pdb; pdb.set_trace()
                print(f'{i} not tested on')
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


def mean_f1(y_true, y_pred, dataset=None, eval=False, dir_path="."):
    test_f1_log_path = os.path.join(dir_path, "average_f1.txt")
    
    # Threshold (max in row = 1; else 0)
    print(f'a few rows of y_pred: {y_pred[-10:]}')
    y_pred = tf.where(
        tf.equal(tf.reduce_max(y_pred, axis=1, keepdims=True), y_pred), 
        tf.constant(1, shape=y_pred.shape), 
        tf.constant(0, shape=y_pred.shape)
    )
    assert(tf.reduce_all(tf.reduce_sum(y_pred, axis=1) == 1))
    with open(test_f1_log_path, "w") as f:
        f1s = []
        for i in range(y_true.shape[1]):
            try:
                num = np.sum(y_true[:, i])

                if num == 0:
                    raise ZeroDivisionError

                print(f'{dataset.unique_labels[i]} # predicted positive: {tf.reduce_sum(y_pred[:, i])}')
                f1 = f1_score(y_true[:, i], y_pred[:, i])
                f1s.append(f1)
            except ZeroDivisionError:
                print(f'{dataset.unique_labels[i]} not tested on')
                f1 = 'N/A'
            if eval:
                f.write(f"{dataset.unique_labels[i]}: {f1}\n")
        mean_f1 = np.mean(f1s)
        
        if eval:
            f.write("-------------------------\n")
            f.write(f"mean f1: {mean_f1}\n")
    if eval:
        print(f"mean f1: {mean_f1}")
    return mean_f1

def proto_sup_acc_outer(num_classes=5, num_samples_per_class=3, num_sup=5):
    def proto_sup_acc(labels, features):
        """
        labels: [n * k + n_query, 2] 
        features: [n * k + n_query, 128]
        """
        support_labels = labels[:num_classes * num_samples_per_class, 0]
        support_labels = support_labels.reshape((num_classes, num_samples_per_class))
        support_features = features[:num_classes * num_samples_per_class]
        support_features = support_features.reshape((num_classes, num_samples_per_class, -1))
            
        prototypes = tf.reduce_mean(support_features, axis=1)
        prototype_labels = tf.reduce_mean(support_labels, axis=1)

        supports = features[:num_classes * num_samples_per_class]
        support_preds = get_nearest_neighbour(supports, prototypes)
        num_correct = np.where(support_preds == labels[:num_classes * num_samples_per_class, 0])[0].shape[0]
        acc = num_correct / num_sup
        return acc
    
    return proto_sup_acc
        
def extract_prototypes_and_queries(num_classes, num_samples_per_class, num_query, labels, features):
    support_labels = labels[:num_classes * num_samples_per_class, 0]
    support_labels = support_labels.reshape((num_classes, num_samples_per_class))
    support_features = features[:num_classes * num_samples_per_class]
    support_features = support_features.reshape((num_classes, num_samples_per_class, -1))

    prototypes = tf.reduce_mean(support_features, axis=1)

    queries = features[num_classes * num_samples_per_class: num_classes * num_samples_per_class + num_query]
    query_labels = labels[num_classes * num_samples_per_class: num_classes * num_samples_per_class + num_query, 0]
    
    return prototypes, queries, query_labels
    
    
def proto_acc_outer(num_classes=5, num_samples_per_class=3, num_query=5):
    def proto_acc(labels, features):
        """
        labels: [n * k + n_query, 2] 
        features: [n * k + n_query, 128]
        """
        prototypes, queries, query_labels = extract_prototypes_and_queries(num_classes, num_samples_per_class,
                                                                                         num_query, labels, features)
        query_preds = get_nearest_neighbour(queries, prototypes)
        num_correct = np.where(query_preds == query_labels)[0].shape[0]
        acc = num_correct / num_query
        return acc
    
    return proto_acc
    
    
def get_query_preds_and_labels(num_classes, num_samples_per_class, num_query, labels, features):

    prototypes, queries, query_labels = extract_prototypes_and_queries(num_classes, num_samples_per_class, 
                                                                       num_query, labels, features)
    
    query_labels_one_hot = np.eye(num_classes)[np.array(query_labels).astype(int)]

    query_distances = get_distances(queries, prototypes)
    query_preds = tf.nn.softmax(-1*query_distances)

    return query_preds, query_labels_one_hot, query_labels


def proto_mean_auroc_outer(num_classes=5, num_samples_per_class=3, num_query=5):
    def proto_mean_auroc(labels, features):
        
        query_preds, query_labels_one_hot, query_labels = get_query_preds_and_labels(num_classes, num_samples_per_class,
                                                                                     num_query, labels, features)
        return mean_auroc(y_true=query_labels_one_hot, y_pred=query_preds)
    
    return proto_mean_auroc


def get_auroc_score_for_class(num_classes, num_samples_per_class, num_query, labels, features, class_idx):
    """
    class_idx: 0 = covid, 1 = tb
    """
    query_preds, query_labels_one_hot, query_labels = get_query_preds_and_labels(num_classes, num_samples_per_class, 
                                                                                 num_query, labels, features)
        
    try:
        mask = tf.cast(labels[-num_query:, class_idx], tf.bool)
        num = np.sum(mask)

        if num == 0:
            raise ZeroDivisionError
        
        idx = int(query_labels[np.array(mask)][0]) ## gets categorical label for covid

        auroc_score = roc_auc_score(query_labels_one_hot[:, idx], query_preds[:, idx])
    except ZeroDivisionError:
        auroc_score = tf.convert_to_tensor([])
    
    return auroc_score


def proto_mean_auroc_covid_outer(num_classes=5, num_samples_per_class=3, num_query=5):
    def proto_mean_auroc_covid(labels, features):
        
        covid_auroc_score = get_auroc_score_for_class(num_classes, num_samples_per_class, num_query, 
                                                      labels, features, class_idx=0)
            
        return covid_auroc_score
    return proto_mean_auroc_covid

def proto_mean_auroc_tb_outer(num_classes=5, num_samples_per_class=3, num_query=5):
    def proto_mean_auroc_tb(labels, features):
        
        tb_auroc_score = get_auroc_score_for_class(num_classes, num_samples_per_class, num_query, 
                                                      labels, features, class_idx=1)
        return tb_auroc_score
    return proto_mean_auroc_tb


def proto_mean_auroc_covid_tb_outer(num_classes=5, num_samples_per_class=3, num_query=5):
    def proto_mean_auroc_covid_tb(labels, features):
        try: 
            covid_mask = tf.cast(labels[-num_query:, 0], tf.bool)
            tb_mask = tf.cast(labels[-num_query:, 1], tf.bool)
            num_covid_tb = np.sum(covid_mask | tb_mask)

            if num_covid_tb == 0:
                raise ZeroDivisionError

            covid_auroc = proto_mean_auroc_covid(num_classes, num_samples_per_class, num_query)
            tb_auroc = proto_mean_auroc_tb(num_classes, num_samples_per_class, num_query)

            print(covid_auroc, tb_auroc)
            auroc = tf.reduce_mean(tf.concat([covid_auroc, tb_auroc], axis=0))
        except ZeroDivisionError:
            # no covid or tb in meta-test task
            auroc = tf.convert_to_tensor([])
        
        return auroc
    return proto_mean_auroc_covid_tb


def proto_mean_f1_outer(num_classes=5, num_samples_per_class=3, num_query=5):
    def proto_mean_f1(labels, features):
    
        query_preds, query_labels_one_hot, query_labels = get_query_preds_and_labels(num_classes, num_samples_per_class,
                                                                                     num_query, labels, features)
    
        return mean_f1(y_true=query_labels_one_hot, y_pred=query_preds)
               
    return proto_mean_f1


def get_f1_for_class(num_classes, num_samples_per_class, num_query, labels, features, class_idx):
    query_preds, query_labels_one_hot, query_labels = get_query_preds_and_labels(num_classes, num_samples_per_class,
                                                                                 num_query, labels, features)

    query_preds = tf.where(
    tf.equal(tf.reduce_max(query_preds, axis=1, keepdims=True), query_preds), 
    tf.constant(1, shape=query_preds.shape), 
    tf.constant(0, shape=query_preds.shape)
    )

    try:
        mask = tf.cast(labels[-num_query:, class_idx], tf.bool)
        num = np.sum(mask)

        if num == 0:
            raise ZeroDivisionError

        idx = int(query_labels[np.array(mask)][0]) ## gets categorical label for covid

        f1 = f1_score(query_labels_one_hot[:, idx], query_preds[:, idx])
    except ZeroDivisionError:
        f1 = tf.convert_to_tensor([])

    return f1


def proto_mean_f1_covid_outer(num_classes=5, num_samples_per_class=3, num_query=5):
    def proto_mean_f1_covid(labels, features):
        covid_f1_score = get_f1_for_class(num_classes, num_samples_per_class, num_query, labels, features, class_idx=0)
            
        return covid_f1_score
    return proto_mean_f1_covid

def proto_mean_f1_tb_outer(num_classes=5, num_samples_per_class=3, num_query=5):
    def proto_mean_f1_tb(labels, features):
        tb_f1_score = get_f1_for_class(num_classes, num_samples_per_class, num_query, labels, features, class_idx=1)
        
        return tb_f1_score
    return proto_mean_f1_tb


def proto_mean_f1_covid_tb_outer(num_classes=5, num_samples_per_class=3, num_query=5):
    def proto_mean_f1_covid_tb(labels, features):
        try: 
            covid_mask = tf.cast(labels[-num_query:, 0], tf.bool)
            tb_mask = tf.cast(labels[-num_query:, 1], tf.bool)
            num_covid_tb = np.sum(covid_mask | tb_mask)

            if num_covid_tb == 0:
                raise ZeroDivisionError

            covid_auroc = proto_mean_f1_covid_outer(num_classes, num_samples_per_class, num_query)
            tb_auroc = proto_mean_f1_tb_outer(num_classes, num_samples_per_class, num_query)

            f1 = tf.reduce_mean(tf.concat([covid_auroc, tb_auroc], axis=0))
        except ZeroDivisionError:
            # no covid or tb in meta-test task
            f1 = tf.convert_to_tensor([])
        
        return f1
    return proto_mean_f1_covid_tb



def get_nearest_neighbour(queries, prototypes):
    """
    queries: [batch_size, embedding_dim]
    prototypes: [num_classes, embedding_dim]

    return:
    categorical preds: (batch_size, )
    """

    distances = get_distances(queries, prototypes)
    pred = np.argmin(distances, axis=1)
    
    return pred ## (batch_size,) (categorical)


def get_distances(queries, prototypes):
    distances = tf.norm(queries[:, None, :] - prototypes[None, :, :], axis=-1)
    
    return distances


def generate_metric_plots(filepath): 
    ''' Generate plots with training and validation metrics. 
        Input filepath specifies path to pickle file storing hist object from training.
    '''
    plt.figure(1, figsize=(5, 5))
    hist_dict = pickle.load(open(filepath, "rb"))
    metrics = ['loss', 'mean_auroc', 'f1_score'] # edit this array if you want to see more metrics
    # Plot train and val metrics 
    for metric in metrics:
        val_metric = 'val_' + metric 
        plt.plot(hist_dict[metric])
        plt.plot(hist_dict[val_metric])
        title = 'Model ' + metric 
        plt.title(title)
        plt.ylabel(metric)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        fname = metric + '.png'
        plt.savefig(fname)
        plt.clf() 
        
def process_tSNE(features, learning_rate=10, perplexity=20, n_iter=1000, early_exaggeration=20):
    """ Computes tNSE embedding as array"""
    tsne = manifold.TSNE(n_components=2, init="random", learning_rate=learning_rate, random_state=0, 
                         perplexity=perplexity, n_iter=n_iter, early_exaggeration=early_exaggeration)
    encoded = tsne.fit_transform(features)
    return encoded

def plot_tsne(tsne_features, tsne_label_strs, plot_title='test', save_path='test.png'):
    """ Plots all given embeddings, but allows for plotting only some classes. """
    
    num_classes = np.unique(tsne_label_strs).shape[0]
    num_samples = tsne_features.shape[0]
    
    tsne_labels_df = pd.DataFrame({'label_str': tsne_label_strs})
    tsne_labels_df['label_num'] = tsne_labels_df.groupby(['label_str']).ngroup()
    print(tsne_labels_df)
    tsne_labels = tsne_labels_df['label_num'].values
    
    label_names = tsne_labels_df.sort_values(by=['label_num'])['label_str'].drop_duplicates().values
    print(label_names)
    
    if label_names is None:
        plt.figure()
        plt.scatter(tsne_features[...,0], tsne_features[...,1], c=tsne_labels, cmap=plt.cm.get_cmap('viridis', num_classes))
        plt.colorbar(ticks=np.arange(num_classes))
        plt.tight_layout()
        plt.savefig(save_path)
    else:
        plt.figure()
        ax = plt.subplot(111)
        for i in range(num_classes): 
            scat = ax.scatter(
                tsne_features[tsne_labels==i,0], tsne_features[tsne_labels==i,1], 
                c=np.repeat(colormap(i, num_classes).reshape(-1,1), [tsne_features[tsne_labels==i,0].shape[0]]).reshape(4,-1).T,
                alpha=0.9,
                label=label_names[i])

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                fancybox=True, shadow=True, ncol=5)

        plt.savefig(save_path)
    return

