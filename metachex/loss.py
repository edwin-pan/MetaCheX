import tensorflow as tf

class Losses():
    
    def __init__(self, class_weights):
        self.class_weights = class_weights

    # def get_weighted_loss(weights):
#     def weighted_loss(y_true, y_pred):
#         return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
#     return weighted_loss
        
    def weighted_binary_crossentropy(self):
        """class_weights: array of size (27, )"""
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
        def weighted_loss(y_true, y_pred):
            return self.class_weights*bce(y_true, y_pred)
        return weighted_loss