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
    
    def __init__(self, class_weights=None, child_to_parent_map=None, train_stage=None,
                 parent_emb_path="training_progress/parent_emb.npy", batch_size=8, 
                 embed_dim=128,
                 parent_multiclass_labels_path=os.path.join(PATH_TO_DATA_FOLDER, 'parent_multiclass_labels.npy'),
                 stage_num=1, parent_weight=0.5, child_weight=0.2, stage2_weight=1.,
                 num_classes=5, num_samples_per_class=3, num_query=5):
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
        self.stage_num = stage_num
        
        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class
        self.num_query = num_query
        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class
        self.num_query = num_query
        
        
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
    
    
    def supcon_label_loss_proto(self, labels, features):
        support_labels = labels[:self.num_classes * self.num_samples_per_class, 0]
        query_labels = labels[-self.num_query:, 0]
        labels = np.concatenate((support_labels, query_labels))

        labels = np.eye(self.num_classes)[labels]

        return supcon_label_loss_inner(labels, features)
   
    
    def supcon_label_loss(self, proto=False):
        """
        features (ie the z's): [batch_size, embedding_dim]
        labels (ie, the y's): [batch_size, num_labels], where labels are one-hot encoded
        """
        
        return self.supcon_label_loss_proto if proto else supcon_label_loss_inner
    
    
    def supcon_class_loss_proto(self, labels, features):
        
        support_labels = labels[:self.num_classes * self.num_samples_per_class, 1]
        query_labels = labels[-self.num_query:, 1]
        labels = np.concatenate((support_labels, query_labels))

        return self.class_contrastive_loss(labels, features, proto=True)

    
    def class_contrastive_loss(self, labels, features, proto=False):
        """
        proto: True iff in protonet framework 
        features (ie the z's): [batch_size, embedding_dim]
        labels (ie, the y's): [batch_size, num_labels], where labels are one-hot encoded iff proto=False; otherwise cat
        Assumes self.embedding_matrix is a 28x128 matrix of embedding vectors, where the ith
        column vector is the embedding of parent with label i.
        
        losses.shape (batch_size, )
        """
        losses = np.zeros(features.shape[0])
        weight = self.parent_weight
        child_weight = self.child_weight
        if self.stage_num == 2:
            # For each example in labels, find index where example[index] == 1
            if proto:
                class_labels = labels
            else:
                class_labels = np.where(labels == 1)[1]
            for i, multiclass_label in enumerate(class_labels):
                ## Note: will never encounter parents that do not exist individually (because not in dataset)
                if multiclass_label in self.parent_multiclass_labels: # Parent label (multiclass) and 'no finding' label
                    ## Get corresponding multitask label
                    multitask_label = np.where(self.parent_multiclass_labels == multiclass_label)[0][0]
                    # Update embedding dict with weighted average of existing embedding and mean batch embedding for label
                    self.embedding_matrix[multitask_label] = weight*self.embedding_matrix[multitask_label] + \
                        (1-weight)*tf.reduce_mean(features[np.where(class_labels==multiclass_label)], axis=0)
                    #losses.append(np.zeros(labels.shape[1])) # No childParent loss for parent
                else: # If child, compute loss with average parent embedding as stored in self.embedding_matrix
                    parent_indices = self.child_to_parent_map[multiclass_label]
                    avg_parent_embeds = tf.reduce_mean(self.embedding_matrix[parent_indices], axis=0)
                    
                    ## Update parent embeddings if parent does not exist by itself
                    parents_no_indiv = parent_indices[self.parent_multiclass_labels[parent_indices] == -1]
                    
                    self.embedding_matrix[parents_no_indiv] = child_weight * self.embedding_matrix[parents_no_indiv] + \
                                                              (1 - child_weight) * features[i]
                    
                    losses[i] = tf.reduce_mean(tf.math.square(avg_parent_embeds - features[i])) # squared loss
            
            losses = tf.convert_to_tensor(losses)
        return tf.zeros(self.batch_size)

        
    def supcon_full_loss(self, proto=False):
        """
        features (ie the z's): [batch_size, embedding_dim]
        labels (ie, the y's): [batch_size, num_labels], where labels are one-hot encoded
        """

        def supcon_class_loss_inner(labels, features):
            loss = self.stage2_weight * self.class_contrastive_loss(labels, features)
            return loss

        def supcon_full_loss_inner(labels, features):
            return supcon_label_loss_inner(labels, features) + supcon_class_loss_inner(labels, features)
        
        def proto_supcon_full_loss_inner(labels, features):
            """
            labels: [n + n_query, 2]; proto-labels: labels[:, 0]; multiclass_labels: labels[:, 1]
            features: [n * k + n_query, 128]
            """
            
            return self.supcon_label_loss_proto(labels, features) + self.supcon_class_loss_proto(labels, features)

        
        return proto_supcon_full_loss_inner if proto else supcon_full_loss_inner
    
    
    def proto_loss(self):
        def proto_loss_inner(labels, features):
            """
            labels: [n * k + n_query, 2]; proto-labels: labels[:, 0]; multiclass_labels: labels[:, 1]
            features: [n * k + n_query, 128]
            """
            support_labels = labels[:self.num_classes * self.num_samples_per_class, 0]
            support_labels = support_labels.reshape((self.num_classes, self.num_samples_per_class, -1))
            support_features = features[:self.num_classes * self.num_samples_per_class]
            support_features = support_features.reshape((self.num_classes, self.num_samples_per_class, -1))
            
            prototypes = tf.reduce_mean(support_features, axis=1)
#             prototype_labels = support_labels
            prototype_labels = tf.reduce_mean(support_labels, axis=1)
            
            queries = features[-self.num_query:]
            query_labels = labels[-self.num_query:, 0]
#             query_labels = np.eye(self.num_classes)[np.array(labels[-self.num_query:, 0])]
            
#             query_distances = tf.norm(queries[:, None, :] - prototypes[None, :, :], axis=-1)
    
            query_distances = get_distances(queries, prototypes)
#             query_distances = tf.convert_to_tensor(query_distances)
            
            ## loss.shape: (n, )
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=-1 * query_distances,
                                                                          labels=tf.stop_gradient(query_labels))
#             loss = tf.reduce_mean(loss)
            
            ## extend to (n * k + n_query, )
#             loss = tf.ones(self.num_classes * self.num_samples_per_class + self.num_query) * loss
#             print(f'loss: {loss}')
            return loss
        
        return proto_loss_inner
            
            
