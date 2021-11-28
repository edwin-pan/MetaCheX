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

def get_sampled_ds(ds, multiclass=True, max_per_class=20):
    
    if multiclass:
        num_classes = ds.num_classes_multiclass
        sampled_df = get_sampled_df_multiclass(ds.df, num_classes=num_classes, max_per_class=max_per_class)
    else:
        num_classes = ds.num_classes_multitask
        sampled_df = get_sampled_df_multitask(ds.df, num_classes=num_classes, max_per_class=max_per_class)
    
    sampled_ds = ImageSequence(df, shuffle_on_epoch_end=False, num_classes=num_classes, multiclass=multiclass)
    
    return sampled_ds


def get_sampled_df_multitask(train_df, num_classes, max_per_class=20):
    """
    Sample max_per_class samples from each (multitask) class in train_df -- repeats are ok
    """
    sampled_df = pd.DataFrame(columns=train_df.columns)
    
    label_multitask_arr = np.array(train_df['label_multitask'].to_list()) ## [len(train_df), 27]
    row_indices, multitask_indices = np.where(label_multitask_arr == 1)

    for i in range(num_classes):
        children_rows = row_indices[multitask_indices == i]
        df_class = train_df.iloc[children_rows]

        if len(df_class) > max_per_class:
            df_class = df_class.sample(max_per_class)

        sampled_df = sampled_df.append(df_class)
    
    sampled_df = sampled_df.reset_index(drop=True)
    return sampled_df

def get_sampled_df_multiclass(train_df, num_classes, parents_only=False, max_per_class=20):
    """
    Sample max_per_class samples from each class in train_df
    if self.parents_only, sample only the parents that exist and the children of parents that don't
    """
    sampled_df = pd.DataFrame(columns=train_df.columns)

    if not parents_only:
        for i in range(num_classes):
            df_class = train_df[train_df['label_num_multi'] == i]

            if len(df_class) > max_per_class:
                df_class = df_class.sample(max_per_class)

            sampled_df = sampled_df.append(df_class)

    else: ## to get parent embedding matrix
        label_multitask_arr = np.array(train_df['label_multitask'].to_list()) ## [len(train_df), 27]
        row_indices, multitask_indices = np.where(label_multitask_arr == 1)

        for i, label in enumerate(self.dataset.parent_multiclass_labels):
            if label != -1: ## Sample parents that exist individually
                df_class = train_df[train_df['label_num_multi'] == label]

            else: ## Sample children of parents that don't exist individually
                ## Get rows where multitask_indices includes i
                children_rows = row_indices[multitask_indices == i]
                df_class = train_df.iloc[children_rows]

            if len(df_class) > max_per_class:
                df_class = df_class.sample(max_per_class)

            df_class['parent_id'] = i ## label with parent class
            sampled_df = sampled_df.append(df_class)

    sampled_df = sampled_df.reset_index(drop=True)

    return sampled_df


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

def mean_auroc(y_true, y_pred, dataset, eval=False, dir_path='.'):
    ## Note: roc_auc_score(y_true, y_pred, average='macro') #doesn't work for some reason -- didn't look into it too much
    aurocs = []
    test_auroc_log_path = os.path.join(dir_path, "auroc.txt")
    with open(test_auroc_log_path, "w") as f:
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


def average_precision(y_true, y_pred, dataset, dir_path="."):
    test_ap_log_path = os.path.join(dir_path, "average_prec.txt")
    with open(test_ap_log_path, "w") as f:
        aps = []
        for i in range(y_true.shape[1]):
            try:
                ap = average_precision_score(y_true[:, i], y_pred[:, i])
                aps.append(ap)
            except RuntimeError:
                ap = 'N/A'
            f.write(f"{dataset.unique_labels[i]}: {ap}\n")
        mean_ap = np.mean(aps)
        f.write("-------------------------\n")
        f.write(f"mean average precision: {mean_ap}\n")

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
        
def process_tSNE(features, learning_rate=10, perplexity=20):
    """ Computes tNSE embedding as array"""
    tsne = manifold.TSNE(n_components=2, init="random", learning_rate=learning_rate, random_state=0, perplexity=perplexity)
    encoded = tsne.fit_transform(features)
    return encoded

def plot_tsne(tsne_features, tsne_labels_one_hot, 
                              label_names=None, 
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

