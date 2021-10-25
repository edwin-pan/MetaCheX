import os
import tensorflow as tf
import numpy as np
from keras import backend as K
    
# This line must be executed before loading Keras model.
# K.set_learning_phase(0)

# def freeze_session(session, keep_var_names=None, output_names=None,clear_devices=True):
#     """
#     Freezes the state of a session into a pruned computation graph.

#     Creates a new computation graph where variable nodes are replaced by
#     constants taking their current value in the session. The new graph will be
#     pruned so subgraphs that are not necessary to compute the requested
#     outputs are removed.
#     @param session The TensorFlow session to be frozen.
#     @param keep_var_names A list of variable names that should not be frozen,
#                           or None to freeze all the variables in the graph.
#     @param output_names Names of the relevant graph outputs.
#     @param clear_devices Remove the device directives from the graph for better portability.
#     @return The frozen graph definition.
#     """
#     from tensorflow.python.framework.graph_util import convert_variables_to_constants
#     graph = session.graph
#     with graph.as_default():
#         freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
#         output_names = output_names or []
#         output_names += [v.op.name for v in tf.global_variables()]
#         # Graph -> GraphDef ProtoBuf
#         input_graph_def = graph.as_graph_def()
#         if clear_devices:
#             for node in input_graph_def.node:
#                 node.device = ""
#         frozen_graph = convert_variables_to_constants(session, input_graph_def,
#                                                       output_names, freeze_var_names)
#         return frozen_graph

def get_model(class_names, weights_path=None, input_shape=None):

    img_input = tf.keras.layers.Input(shape=input_shape)
    base_model = tf.keras.applications.densenet.DenseNet121(include_top=False, weights=None, 
                                                            input_tensor=img_input, pooling='avg')


    x = base_model.output
    predictions = tf.keras.layers.Dense(len(class_names), activation="sigmoid", name="predictions")(x)
    model = tf.keras.models.Model(inputs=img_input, outputs=predictions)
    
    if weights_path == "":
        weights_path = None

    if weights_path is not None:
        print(f"load model weights_path: {weights_path}")
        model.load_weights(weights_path)

    return model


def save_model():
    output_dir = '.'
    output_weights_name = 'chexnet_weights.h5'
    model_weights_file = os.path.join(output_dir, output_weights_name)

    base_model = get_model(np.arange(14), 
                           weights_path=model_weights_file,
                           input_shape=(224, 224, 3))
    
#     base_model.save_weights('chexnet_weights')
    base_model.save('chexnet')

   
if __name__ == '__main__':
    save_model()
