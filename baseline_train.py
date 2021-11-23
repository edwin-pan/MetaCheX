import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# metachex materials
from metachex.configs.config import *
from metachex.dataloader import MetaChexDataset
from metachex.loss import Losses
from metachex.utils import process_tSNE, plot_tsne

def parse_args():
    parser = argparse.ArgumentParser(description='Baseline MetaChex: Fine-Tuned ChexNet')
    parser.add_argument('-c', '--ckpt_save_path', default='training_progress/cp_best.ckpt')
    parser.add_argument('-t', '--tsne', action='store_true', help='Generate tSNE plot')
    parser.add_argument('-e', '--evaluate', action='store_true', help='Evaluate model performance')
    parser.add_argument('-p', '--pretrained', default=None, help='Path to pretrained weights, if desired')
    parser.add_argument('-n', '--num_epochs', default=15, help='Number of epochs to train for')
    return parser.parse_args()





def load_chexnet_pretrained(class_names=np.arange(14), weights_path='chexnet_weights.h5', 
                            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)):

    img_input = tf.keras.layers.Input(shape=input_shape)
    base_model = tf.keras.applications.densenet.DenseNet121(include_top=False, weights=None, 
                                                            input_tensor=img_input, pooling='avg')
    base_model.trainable = False


    x = base_model.output
    predictions = tf.keras.layers.Dense(len(class_names), activation="sigmoid", name="predictions")(x)
    model = tf.keras.models.Model(inputs=img_input, outputs=predictions)
    model.load_weights(weights_path)

    return model


def load_chexnet(output_dim, embedding_dim=128):
    """
    output_dim: dimension of output
    """
    
    base_model_old = load_chexnet_pretrained()
    x = base_model_old.layers[-2].output ## remove old prediction layer
    
    ## The prediction head can be more complicated if you want
    embeddings = tf.keras.layers.Dense(embedding_dim, name='embedding', activation='relu')(x)
    normalized_embeddings = tf.nn.l2_normalize(embeddings, axis=-1)
    predictions = tf.keras.layers.Dense(output_dim, name='prediction', activation='sigmoid')(normalized_embeddings) # BASELINE: directly predict
    chexnet = tf.keras.models.Model(inputs=base_model_old.inputs,outputs=predictions)
    return chexnet
    #base_model_old.trainable=False
    #return base_model_old

def get_embedding_model(model):
    x = model.layers[-2].output
    chexnet_embedder = tf.keras.models.Model(inputs = model.input, outputs = x)
    return chexnet_embedder

def mean_auroc(y_true, y_pred, eval=False):
    ## Note: roc_auc_score(y_true, y_pred, average='macro') #doesn't work for some reason -- didn't look into it too much
    aurocs = []
    with open("test_log.txt", "w") as f:
        for i in range(y_true.shape[1]):
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                aurocs.append(score)
            except ValueError:
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


def average_precision(y_true, y_pred, class_names):
    test_ap_log_path = os.path.join(".", "average_prec.txt")
    with open(test_ap_log_path, "w") as f:
        aps = []
        for i in range(len(class_names)):
            ap = average_precision_score(y_true[:, i], y_pred[:, i])
            aps.append(ap)
            f.write(f"{class_names[i]}: {ap}\n")
        mean_ap = np.mean(aps)
        f.write("-------------------------\n")
        f.write(f"mean average precision: {mean_ap}\n")



def train(num_epochs=15, checkpoint_path="training_progress/cp_best.ckpt"):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    monitor='val_mean_auroc',
                                                    mode='max',
                                                    save_best_only=True)

    hist = chexnet.fit(dataset.train_ds,
                validation_data=dataset.val_ds,
                epochs=num_epochs,
                steps_per_epoch=dataset.train_steps_per_epoch, ## size(train_ds) * 0.125 * 0.1
                validation_steps=dataset.val_steps_per_epoch, ## size(val_ds) * 0.125 * 0.2
                batch_size=dataset.batch_size, ## 8
                # class_weight=class_weights,
                callbacks=[cp_callback]
                )

    with open(os.path.join(checkpoint_dir, 'trainHistoryDict'), 'wb') as file_pi:
            pickle.dump(hist.history, file_pi)

    return hist


def compile():
    class_weights, indiv_class_weights, _ = dataset.get_class_weights(one_cap=True)

    loss_fn = Losses(class_weights, batch_size=dataset.batch_size)

    # output_dim = len(unique_labels)
    chexnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                    loss=loss_fn.weighted_binary_crossentropy(),
    #                   loss_weights=1e5,
    #                 loss='binary_crossentropy',
                    #metrics=[tf.keras.metrics.AUC(multi_label=True),  
                    metrics=[mean_auroc, #mean_precision, mean_recall, 'binary_accuracy', 'accuracy', 
                            tfa.metrics.F1Score(average='micro',num_classes=dataset.num_classes_multitask)], 
                    #        tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
                    run_eagerly=True)


if __name__ == '__main__':
    args = parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"]=""
    tf.test.is_gpu_available()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Instantiate dataset
    dataset = MetaChexDataset()
    
    # Grab training dataset
    train_ds = dataset.train_ds

    # Load CheXNet
    chexnet = load_chexnet(dataset.num_classes_multitask)
    chexnet.trainable = True
    
    print(chexnet.summary())
    
    # Compile
    compile()

    # Get weights
    if args.pretrained is None:
        print("[INFO] Beginning Fine Tuning")
        # Train
        hist = train(args.num_epochs, args.ckpt_save_path)
        record_dir = os.path.dirname(args.ckpt_save_path)
    else:
        print("[INFO] Loading weights")
        # Load weights
        chexnet.load_weights(args.pretrained)
        record_dir = os.path.dirname(args.pretrained)

    # Evaluate
    if args.evaluate:
        print("[INFO] Evaluating performance")
        y_test_true = dataset.test_ds.get_y_true() 
        y_test_pred = chexnet.predict(dataset.test_ds, verbose=1)
        mean_auroc(y_test_true, y_test_pred, eval=True)
        average_precision(y_test_true, y_test_pred)

    # Generate tSNE
    if args.tsne:
        print("[INFO] Generating tSNE plots")
        chexnet_embedder = get_embedding_model(chexnet)
        tsne_dataset = MetaChexDataset(shuffle_train=False)

        embedding_save_path = os.path.join(record_dir, 'embeddings.npy')
        # generating embeddings can take some time. Load if possible
        if os.path.isfile(embedding_save_path):
            print(f"[INFO] Embeddings already processed. Loading from {embedding_save_path}")
            training_embeddings = np.load(embedding_save_path)
        else:
            print(f"[INFO] Embeddings processing. Saving to {embedding_save_path}")
            training_embeddings = chexnet_embedder.predict(tsne_dataset.train_ds, verbose=1)
            np.save(embedding_save_path, training_embeddings)

        tsne_feats = process_tSNE(training_embeddings)
        tsne_labels = tsne_dataset.train_ds.get_y_true()

        plot_tsne(tsne_feats, tsne_labels, lables_names=tsne_dataset.unique_labels)
