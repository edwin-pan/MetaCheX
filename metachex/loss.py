import tensorflow as tf
import numpy as np

class Losses():
    
    def __init__(self, class_weights, batch_size=8):
        self.class_weights = class_weights
        self.batch_size = batch_size
    # def get_weighted_loss(weights):
#     def weighted_loss(y_true, y_pred):
#         return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
#     return weighted_loss
        
    def weighted_binary_crossentropy(self):
        """class_weights: array of size (27, )"""
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    
        def weighted_loss(y_true, y_pred):
            
            if y_true.shape[0] != None:
                new_weight = np.zeros(y_true.shape)
                trues_x_idx, trues_z_idx = np.where(y_true==1)
                false_x_idx, false_z_idx = np.where(y_true==0)
                new_weight[(trues_x_idx, trues_z_idx)] = self.class_weights[1][trues_z_idx]
                new_weight[(false_x_idx, false_z_idx)] = 0 # self.class_weights[0][false_z_idx]
            else:
                new_weight = self.class_weights[0] # Never use this. Hacky way to get around TF

            y_true_fat = y_true.reshape([-1, y_true.shape[1], 1])
            y_pred_fat = y_pred.reshape([-1, y_pred.shape[1], 1])
            return new_weight*bce(y_true_fat, y_pred_fat)
        return weighted_loss