import tensorflow as tf
import numpy as np
import sys
import pickle
import os
from metachex.configs.config import *
from metachex.utils import get_distances

sys.path.append('../') # importing in unit tests
from supcon.losses import contrastive_loss

def supcon_label_loss_inner(labels, features):
    """
    loss.shape (batch_size, )
    """
    features = tf.expand_dims(features, axis=1)
    loss = contrastive_loss(features, labels)
    return loss


class Losses():
    
    def __init__(self, class_weights=None, num_classes=5, num_samples_per_class=3, num_query=5):
        
        self.class_weights = class_weights
        
        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class
        self.num_query = num_query
        
        
    def weighted_binary_crossentropy(self):
        """class_weights: array of size (18, )"""
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

            return new_weight*bce(y_true_fat, y_pred_fat)
        return weighted_loss
    
    
    def supcon_label_loss_proto(self, labels, features):
        support_labels = labels[:self.num_classes * self.num_samples_per_class, 0]
        query_labels = labels[self.num_classes * self.num_samples_per_class: self.num_classes * self.num_samples_per_class + self.num_query, 0]
        labels = tf.concat([support_labels, query_labels], 0)
        labels = tf.one_hot(labels, self.num_classes)

        return supcon_label_loss_inner(labels, features)
    
    
    def supcon_label_loss(self, proto=False):
        """
        features (ie the z's): [batch_size, embedding_dim]
        labels (ie, the y's): [batch_size, num_labels], where labels are one-hot encoded
        """
        def supcon_label_loss_inner2(labels, features):
            return self.supcon_label_loss_proto(labels, features) if proto else supcon_label_loss_inner(labels, features)
        return supcon_label_loss_inner2    
    
    
    def proto_loss_inner(self, labels, features):
        """
        labels: [n * k + n_query, 2]; proto-labels: labels[:, 0]; multiclass_labels: labels[:, 1]
        features: [n * k + n_query, 128]
        """
        support_labels = labels[:self.num_classes * self.num_samples_per_class, 0]
        support_labels = support_labels.reshape((self.num_classes, self.num_samples_per_class, -1))
        support_features = features[:self.num_classes * self.num_samples_per_class]
        support_features = support_features.reshape((self.num_classes, self.num_samples_per_class, -1))

        prototypes = tf.reduce_mean(support_features, axis=1)

        queries = features[self.num_classes * self.num_samples_per_class: self.num_classes * self.num_samples_per_class + self.num_query]
        query_labels = labels[self.num_classes * self.num_samples_per_class: self.num_classes * self.num_samples_per_class + self.num_query, 0] # 0 indexes proto-label

        query_distances = get_distances(queries, prototypes)

        ## loss.shape: (n, )
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=-1 * query_distances,
                                                                      labels=tf.stop_gradient(query_labels))
        return loss
    
    
    def proto_loss(self):
        def proto_loss_inner2(labels, features):
            return self.proto_loss_inner(labels, features)
        
        return proto_loss_inner2
            
            
