import tensorflow as tf
import numpy as np
import sys

sys.path.append('../') # importing in unit tests
from supcon.losses import contrastive_loss

class Losses():
    
    def __init__(self, class_weights=None, child_to_parent_map=None, batch_size=8):
        """
        child_to_parent_map: mapping of multiclass labels to a list of their parents
                format: {(child multiclass label (int), child label_str) : list[parent multiclass labels (int)]}
        """
        
        self.class_weights = class_weights
        self.batch_size = batch_size
        
        self.child_to_parent_map = child_to_parent_map
        
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
    
    
    def supcon_label_loss(self):
        """
        features (ie the z's): [batch_size, embedding_dim]
        labels (ie, the y's): [batch_size, num_labels], where labels are one-hot encoded
        """
        
        def supcon_label_loss_inner(labels, features):
            features = tf.expand_dims(features, axis=1)
            return contrastive_loss(features, labels)
        
        return supcon_label_loss_inner


    def supcon_class_loss(self)
        """
        features (ie the z's): [batch_size, embedding_dim]
        labels (ie, the y's): [batch_size, num_labels], where labels are one-hot encoded (multiclass)
        """
        
        def supcon_class_loss_inner(labels, features):
            return class_contrastive_loss(self, features, labels)
        
        return supcon_class_loss_inner

    
    def class_contrastive_loss(self, features, labels):
        ## TODO
        # TODO: Implement vectorized-dotprod for measuring how "in-the-middle" the child is
