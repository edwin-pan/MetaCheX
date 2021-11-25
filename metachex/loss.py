import tensorflow as tf
import numpy as np
import sys
import pickle

sys.path.append('../') # importing in unit tests
from supcon.losses import contrastive_loss

class Losses():
    
    def __init__(self, class_weights=None, label_map=None, train_stage=None, emb_path="training_progress/parent_emb.pkl", batch_size=8):
        """
        child_to_parent_map: mapping of multiclass labels to a list of their parents
                format: {(child multiclass label (int), child label_str) : list[parent multiclass labels (int)]}
        """
        self.embedding_map = None
        self.parents_indices = dict(zip(list(range(0,27)), [None]*27)) # indices corresponding to parents
        if train_stage == 1: # Save parent embeddings [Need to save with callback]
            self.embedding_map = dict(zip(list(range(0,27)), [None]*27))
            
        elif train_stage == 2: # Load & update embedding_map 
            with open(emb_path, 'rb') as handle:
                self.embedding_map = pickle.load(handle)
            
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
        
        losses, class_labels = [], []
        weight = 0.5
        
        if self.train_stage == 2:
            # For each example in labels, find index where example[index] == 1
            for i in range(0, labels.shape[0]): 
                class_labels.append(np.where(labels[i]==1)) # no vectorization since we care about individual label vectors
                
            class_labels = np.array(class_labels)
            for i, label in enumerate(0, class_labels):
                if label in self.parents_indices: # if parent label, update embedding dict with mean in batch
                    self.embedding_map[label] = weight*self.embedding_map[label] + \
                        (1-weight)*tf.reduce_mean(features[np.where(class_labels=label)], axis=-1)
                    losses.append(np.zeros(labels[0].shape))
                    
                else: # if child, compute loss with average parent embedding
                    parent_mask = np.in1d(class_labels, self.label_map[label]) # marks True where class_labels has parent_label
                    avg_parent_emb = tf.reduce_mean(features[np.where(parent_mask==True)]) 
                    losses.append(tf.math.square(avg_parent_emb - features[i])) # squared loss
            
            losses = tf.convert_to_tensor(losses)
            return losses 
        else:
            return tf.zeros(features.shape)
        
        
        # for children in batch, find parent embeddings and compute average embedding + loss v
        ## TODO
        # TODO: Implement vectorized-dotprod for measuring how "in-the-middle" the child is
