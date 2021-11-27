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
    
    loss = contrastive_loss(features, labels)
    return loss


class Losses():
    
    def __init__(self, class_weights=None, child_to_parent_map=None, train_stage=None,
                 parent_emb_path="training_progress/parent_emb.npy", batch_size=8, 
                 embed_dim=128,
                 parent_multiclass_labels_path=os.path.join(PATH_TO_DATA_FOLDER, 'parent_multiclass_labels.npy'),
                 stage_num=1, parent_weight=0.5, child_weight=0.2, stage2_weight=1.):
        """
        child_to_parent_map: mapping of multiclass labels to a list of their parents
                format: {child multiclass label (int) : list[parent multitask indices (int)]}
                e.g: {122: array([2, 6])}
        embedding_matrix: [27 x 128] -- parent labels are in alphabetical order
        """
        if child_to_parent_map is not None: ## includes childParent
            self.embedding_matrix = np.load(parent_emb_path) ## [27 x 128]
        
        self.parent_weight = parent_weight
        self.child_weight = child_weight
        self.stage2_weight = stage2_weight
        
        ## each entry corresponds to multiclass label for that multitask index
        ## parents who do not exist individually will be marked by a -1 in the self.parent_multiclass_labels array
        self.parent_multiclass_labels = np.load(parent_multiclass_labels_path) ## (27, ) 
        
        if child_to_parent_map is not None:
            self.child_indices = child_to_parent_map.keys()
            
        self.class_weights = class_weights
        self.batch_size = batch_size
        
        self.child_to_parent_map = child_to_parent_map 
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

            return new_weight*bce(y_true_fat, y_pred_fat)
        return weighted_loss
    
    
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
        weight = self.parent_weight
        child_weight = self.child_weight
        depth = self.parent_multiclass_labels.shape[0] ## 27
        if self.stage_num == 2:
            # For each example in labels, find index where example[index] == 1
            class_labels = np.where(labels == 1)[1]
            for i, multiclass_label in enumerate(class_labels):
                ## Note: will never encounter parents that do not exist individually (because not in dataset)
                if multiclass_label in self.parent_multiclass_labels: # Parent label (multiclass)
                    ## Get corresponding multitask label
                    multitask_label = np.where(self.parent_multiclass_labels == multiclass_label)[0][0]
                    # Update embedding dict with weighted average of existing embedding and mean batch embedding for label
                    self.embedding_matrix[multitask_label] = weight*self.embedding_matrix[multitask_label] + \
                        (1-weight)*tf.reduce_mean(features[np.where(class_labels==multiclass_label)], axis=0)
                    losses.append(np.zeros(labels.shape[1])) # No childParent loss for parent
                else: # If child, compute loss with average parent embedding as stored in self.embedding_matrix
                    parent_indices = self.child_to_parent_map[multiclass_label]
                    avg_parent_embeds = tf.reduce_mean(self.embedding_matrix[parent_indices], axis=0)
                    
                    ## Update parent embeddings if parent does not exist by itself
                    parents_no_indiv = parent_indices[self.parent_multiclass_labels[parent_indices] == -1]
                    
                    self.embedding_matrix[parents_no_indiv] = child_weight * self.embedding_matrix[parents_no_indiv] + \
                                                              (1 - child_weight) * features[i]
                    
                    losses.append(tf.reduce_mean(tf.math.square(avg_parent_embeds - features[i]), axis=1)) # squared loss

            losses = tf.convert_to_tensor(losses)
            return losses
        else:
            return tf.zeros(self.batch_size)

        
    def supcon_full_loss(self):
        """
        features (ie the z's): [batch_size, embedding_dim]
        labels (ie, the y's): [batch_size, num_labels], where labels are one-hot encoded
        """

        def supcon_class_loss_inner(labels, features):
            loss = self.stage2_weight * self.class_contrastive_loss(labels, features)
            return loss

        def supcon_full_loss_inner(labels, features):
            return supcon_label_loss_inner(labels, features) + supcon_class_loss_inner(labels, features)

        return supcon_full_loss_inner