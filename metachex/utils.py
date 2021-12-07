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


def load_chexnet_pretrained(class_names=np.arange(14), weights_path='chexnet_weights.h5', 
                            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)):

    img_input = tf.keras.layers.Input(shape=input_shape)
    base_model = tf.keras.applications.densenet.DenseNet121(include_top=False, #weights=None,  # use imagenet weights
                                                            input_tensor=img_input, pooling='avg')
    base_model.trainable = False


    x = base_model.output
    predictions = tf.keras.layers.Dense(len(class_names), activation="sigmoid", name="predictions")(x)
    model = tf.keras.models.Model(inputs=img_input, outputs=predictions)
#     model.load_weights(weights_path) ## uncomment

    return model

def baby_conv(input_obj):
    model = tf.keras.models.Sequential()
    model.add(input_obj)
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(244, 244, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    # model.add(tf.keras.layers.Dense(embedding_dim))
    # model.add(tf.keras.layers.Dense(1)) # this is not used
    return model

def load_chexnet(output_dim, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), multiclass=False, embedding_dim=128):
    """
    output_dim: dimension of output
    multiclass: whether or not prediction task is multiclass (vs. binary multitask)
    Note: multiclass argument is only relevant for baseline models
    """
    
#     base_model_old = load_chexnet_pretrained()
#     x = base_model_old.layers[-2].output ## remove old prediction layer
    
    img_input = tf.keras.layers.Input(shape=input_shape)
    # base_model = tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet',  # use imagenet weights
    #                                                         input_tensor=img_input, pooling='avg')

    # base_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', 
    #                                                         input_tensor=img_input, pooling='avg')

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
    #base_model_old.trainable=False
    #return base_model_old

def get_embedding_model(model):
    x = model.layers[-2].output
    chexnet_embedder = tf.keras.models.Model(inputs = model.input, outputs = x)
    return chexnet_embedder

def get_sampled_ds(ds, multiclass=True, max_per_class=20):
    
    num_classes = ds.num_classes
    if multiclass:
        sampled_df = get_sampled_df_multiclass(ds.df, num_classes=num_classes, max_per_class=max_per_class)
    else:
        sampled_df = get_sampled_df_multitask(ds.df, num_classes=num_classes, max_per_class=max_per_class)
    
    sampled_ds = ImageSequence(sampled_df, shuffle_on_epoch_end=False, num_classes=num_classes, multiclass=multiclass)
    
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
                import pdb; pdb.set_trace()
                print(f'{dataset.unique_labels[i]} not tested on')
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


def average_precision(y_true, y_pred, dataset, dir_path=".", plot=True):
    
    test_ap_log_path = os.path.join(dir_path, "average_prec.txt")
    with open(test_ap_log_path, "w") as f:
        aps = []
        for i in range(y_true.shape[1]):
            try:
                ap = average_precision_score(y_true[:, i], y_pred[:, i])
                if plot:
                    pr_plot_dir = os.path.join(dir_path, 'pr_plots')
                    os.makedirs(pr_plot_dir, exist_ok=True)
                    precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
                    display = PrecisionRecallDisplay(recall=recall, precision=precision, average_precision=ap)
                    display.plot()
                    plot = display.figure_
                    plot.savefig(os.path.join(pr_plot_dir, f"{dataset.unique_labels[i]}_pr_curve.png"))
                    plt.close()
                aps.append(ap)
            except RuntimeWarning:
                print(f'{dataset.unique_labels[i]} not tested on')
                ap = 'N/A'
            f.write(f"{dataset.unique_labels[i]}: {ap}\n")
        mean_ap = np.mean(aps)
        f.write("-------------------------\n")
        f.write(f"mean average precision: {mean_ap}\n")
    print(f"mean average precision: {mean_ap}")


def mean_f1(y_true, y_pred, dataset=None, eval=False, dir_path="."):
    test_f1_log_path = os.path.join(dir_path, "average_f1.txt")
    
    # Threshold (max in row = 1; else 0)
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
                f1 = f1_score(y_true[:, i], y_pred[:, i])
                f1s.append(f1)
            except RuntimeWarning:
                print(f'{dataset.unique_labels[i]} not tested on')
                f1 = 0
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
            
        # queries = features[-5:]
        # query_labels = labels[-5:, 0]
        # query_preds = get_nearest_neighbour(queries, prototypes)

        supports = features[:num_classes * num_samples_per_class]
        support_preds = get_nearest_neighbour(supports, prototypes)
        num_correct = np.where(support_preds == labels[:num_classes * num_samples_per_class, 0])[0].shape[0]
        acc = num_correct / num_sup
#         print(f"acc: {acc}")
        return acc
    
    return proto_sup_acc
    
def proto_acc_outer(num_classes=5, num_samples_per_class=3, num_query=5):
    def proto_acc(labels, features):
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
            
        queries = features[-num_query:]
        query_labels = labels[-num_query:, 0]
#         print(f"ground truth: \n {query_labels}")
#         query_labels_cat = np.where(query_labels == 1)[1] ## get categorical labels
            
        query_preds = get_nearest_neighbour(queries, prototypes)
        num_correct = np.where(query_preds == query_labels)[0].shape[0]
        acc = num_correct / num_query
#         print(f"acc: {acc}")
        return acc
    
    return proto_acc


def proto_mean_auroc_outer(num_classes=5, num_samples_per_class=3, num_query=5):
    def proto_mean_auroc(labels, features):
        support_labels = labels[:num_classes * num_samples_per_class, 0]
        support_labels = support_labels.reshape((num_classes, num_samples_per_class))
        support_features = features[:num_classes * num_samples_per_class]
        support_features = support_features.reshape((num_classes, num_samples_per_class, -1))
            
        prototypes = tf.reduce_mean(support_features, axis=1)
        prototype_labels = tf.reduce_mean(support_labels, axis=1)
            
        queries = features[-num_query:]
        query_labels = labels[-num_query:, 0]
        query_labels_one_hot = np.eye(num_classes)[np.array(query_labels).astype(int)]
        
        query_distances = get_distances(queries, prototypes)
#         print(f"distances: \n {query_distances}")
        query_preds = tf.nn.softmax(-1*query_distances)
#         print(f"soft predictions: \n {query_preds}")
#         print(f'query_labels: {query_labels}')
#         print(f'query_preds: {query_preds}')
        
        return mean_auroc(y_true=query_labels_one_hot, y_pred=query_preds)
#         return roc_auc_score(y_true=query_labels, y_score=query_preds, 
#                              multi_class='ovr', labels=np.arange(num_classes))
    
    return proto_mean_auroc


def proto_mean_f1_outer(num_classes=5, num_samples_per_class=3, num_query=5):
    def proto_mean_f1(labels, features):
    
        support_labels = labels[:num_classes * num_samples_per_class, 0]
        support_labels = support_labels.reshape((num_classes, num_samples_per_class))
        support_features = features[:num_classes * num_samples_per_class]
        support_features = support_features.reshape((num_classes, num_samples_per_class, -1))
            
        prototypes = tf.reduce_mean(support_features, axis=1)
        prototype_labels = tf.reduce_mean(support_labels, axis=1)
            
        queries = features[-num_query:]
        query_labels = labels[-num_query:, 0]
        query_labels_one_hot = np.eye(num_classes)[np.array(query_labels).astype(int)]
        
        query_distances = get_distances(queries, prototypes)
        query_preds = tf.nn.softmax(-1*query_distances)
    
        return mean_f1(y_true=query_labels_one_hot, y_pred=query_preds)
               
    return proto_mean_f1


def get_nearest_neighbour(queries, prototypes):
    """
    queries: [batch_size, embedding_dim]
    prototypes: [num_classes, embedding_dim]

    return:
    categorical preds: (batch_size, )
    """

    distances = get_distances(queries, prototypes)
    pred = np.argmin(distances, axis=1)
<<<<<<< HEAD
#     print(f"actual predictions: {pred}")
=======
>>>>>>> 96554fd1047158f498572e8880c3d45fa237e1d2
    
    return pred ## (batch_size,) (categorical)
    #return np.eye(prototypes.shape[0])[pred] ## one-hot


def get_distances(queries, prototypes):
#     distances = np.linalg.norm(queries[:, None, :] - prototypes[None, :, :], axis=-1)
    distances = tf.norm(queries[:, None, :] - prototypes[None, :, :], axis=-1)
    
#     distances = euclidean_distances(queries, prototypes)
    
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
    """ Plots all given embeddings, but allows for plotting only some classes. """
    tsne_labels = np.argmax(tsne_labels_one_hot, axis=-1)
    num_classes = tsne_labels_one_hot.shape[-1]
    num_samples = tsne_labels_one_hot.shape[0]

    if visualize_class_list is not None:
        # TODO: Visualize subset of classes
        pass

    if label_names is None:
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

