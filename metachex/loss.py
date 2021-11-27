import tensorflow as tf
import numpy as np
import sys
import pickle
import os
from metachex.configs.config import *

sys.path.append('../') # importing in unit tests
from supcon.losses import contrastive_loss

def supcon_label_loss_inner(labels, features):
    features = tf.expand_dims(features, axis=1)
    return contrastive_loss(features, labels)


class Losses():
    
    def __init__(self, class_weights=None, child_to_parent_map=None, train_stage=None,
                 emb_path="training_progress/parent_emb.pkl", batch_size=8, 
                 num_indiv_parents=27, embed_dim=128,
                 parent_multiclass_labels_path=os.path.join(PATH_TO_DATA_FOLDER, 'parent_multiclass_labels.npy'),
                 stage_num=1, childparent_lambda=1):
        """
        child_to_parent_map: mapping of multiclass labels to a list of their parents
                format: {child multiclass label (int) : list[parent multitask indices (int)]}
                e.g: {122: array([2, 6])}
        """
        self.num_indiv_parents = num_indiv_parents # store depth for creating one-hot labels on fly
        self.embedding_matrix = np.zeros((num_indiv_parents, embed_dim)) ## [27 x 128] -- parent labels are in alphabetical order
        
        ## each entry corresponds to multiclass label for that multitask index
        self.parent_multiclass_labels = np.load(parent_multiclass_labels_path) ## (27, ) 
        
        if child_to_parent_map is not None:
            self.child_indices = child_to_parent_map.keys()
#         if train_stage == 1: # Save parent embeddings [Need to save with callback]
#             self.embedding_map = #dict(zip(list(range(0,27)), [None]*27))
            
#         elif train_stage == 2: # Load & update embedding_map 
#             with open(emb_path, 'rb') as handle:
#                 self.embedding_map = pickle.load(handle)
            
        self.class_weights = class_weights
        self.batch_size = batch_size
        
        self.child_to_parent_map = child_to_parent_map 
        self.childparent_lambda = childparent_lambda
        self.stage_num = train_stage
        
    def weighted_binary_crossentropy(self):
        """class_weights: array of size (27, )"""
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    
        def weighted_loss(y_true, y_pred):
            if y_true.shape[0] != None:
                new_weight = np.zeros(y_true.shape)
                trues_x_idx, trues_z_idx = np.where(y_true==1)
                false_x_idx, false_z_idx = np.where(y_true==0)
                new_weight[(trues_x_idx, trues_z_idx)] = self.class_weights[1][trues_z_idx]
                new_weight[(false_x_idx, false_z_idx)] = self.class_weights[0][false_z_idx]
            else:
                new_weight = self.class_weights[0] # Never use this. Hacky way to get around TF

            y_true_fat = y_true.reshape([-1, y_true.shape[1], 1])
            y_pred_fat = y_pred.reshape([-1, y_pred.shape[1], 1])

            # print(y_true_fat.shape, y_pred_fat.shape)
            return new_weight*bce(y_true_fat, y_pred_fat)
        return weighted_loss
    
    
    def supcon_full_loss(self):
        """
        features (ie the z's): [batch_size, embedding_dim]
        labels (ie, the y's): [batch_size, num_labels], where labels are one-hot encoded
        """
        
        def supcon_class_loss_inner(labels, features):
            return class_contrastive_loss(self, labels, features)
        
        return supcon_label_loss_inner + self.childparent_lambda * supcon_class_loss_inner
    
    
    def supcon_label_loss(self):
        """
        features (ie the z's): [batch_size, embedding_dim]
        labels (ie, the y's): [batch_size, num_labels], where labels are one-hot encoded
        """
        
        return supcon_label_loss_inner


    def class_contrastive_loss(self, labels, features):
        """
        features (ie the z's): [batch_size, embedding_dim]
        labels (ie, the y's): [batch_size, num_labels], where labels are one-hot encoded
        Assumes self.embedding_matrix is a 27x128 matrix of embedding vectors, where the ith
        column vector is the embedding of parent with label i.
        """
        losses, class_labels = [], []
        weight = self.childparent_lambda
        depth = self.num_indiv_parents
        if self.stage_num == 2:
            # For each example in labels, find index where example[index] == 1
            class_labels = np.where(labels == 1)[1]
            for i, label in enumerate(0, class_labels):
                depth = 27 # embeding dimension
                if label >= 0 and label <= 26: # Parent label
                    # Update embedding dict with weighted average of existing embedding and mean batch embedding for label
                    one_hot_label = tf.one_hot(label, depth)
                    self.embedding_matrix[label] = weight*one_hot_label.dot(self.embedding_matrix) + \
                        (1-weight)*tf.reduce_mean(features[np.where(class_labels=label)], axis=0)
                    losses.append(np.zeros(labels[0].shape)) # No childParent loss for parent

                else: # If child, compute loss with average parent embedding as stored in self.embedding_matrix
                    depth = self.embedding_matrix.shape[0] #
                    one_hot_parents = tf.one_hot(self.child_to_parent_map[label], depth)
                    avg_parent_emb = tf.reduce_mean(one_hot_parents.dot(self.embedding_matrix))
                    losses.append(tf.math.square(avg_parent_emb - features[i])) # squared loss

            losses = tf.convert_to_tensor(losses)
            return losses
        else:
            return tf.zeros(features.shape)


# for children in batch, find parent embeddings and compute average embedding + loss v
## TODO
# TODO: Implement vectorized-dotprod for measuring how "in-the-middle" the child is