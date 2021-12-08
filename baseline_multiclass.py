import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# metachex materials
from metachex.configs.config import *
from metachex.dataloader import MetaChexDataset
from metachex.loss import Losses
from metachex.utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Baseline MetaChex: Fine-Tuned ChexNet')
    parser.add_argument('-t', '--tsne', action='store_true', help='Generate tSNE plot')
    parser.add_argument('-e', '--evaluate', action='store_true', help='Evaluate model performance')
    parser.add_argument('-c', '--ckpt_save_path', default='training_progress_baseline_multiclass/cp_best.ckpt')
    parser.add_argument('-p', '--pretrained', default=None, help='Path to pretrained weights, if desired')
    parser.add_argument('-n', '--num_epochs', type=int, default=15, help='Number of epochs to train for')
    return parser.parse_args()


def train(num_epochs=15, checkpoint_path="training_progress_baseline_multiclass/cp_best.ckpt"):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    monitor='val_mean_auroc_baseline',
                                                    mode='max',
                                                    save_best_only=True)

    hist = chexnet.fit(dataset.train_ds,
                validation_data=dataset.val_ds,
                epochs=num_epochs,
                steps_per_epoch=dataset.train_steps_per_epoch, 
                validation_steps=dataset.val_steps_per_epoch,
                batch_size=dataset.batch_size, 
                callbacks=[cp_callback]
                )

    with open(os.path.join(checkpoint_dir, 'trainHistoryDict'), 'wb') as file_pi:
            pickle.dump(hist.history, file_pi)

    return hist


def compile():
    class_weights = dataset.get_class_weights2(one_cap=False)

    loss_fn = Losses(class_weights, batch_size=dataset.batch_size)

    chexnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=loss_fn.weighted_binary_crossentropy(),
                    metrics=[mean_auroc_baseline, 
                             tfa.metrics.F1Score(average=None,num_classes=dataset.num_classes_multiclass)],
                    run_eagerly=True)


if __name__ == '__main__':
    args = parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"]=""
    tf.test.is_gpu_available()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Instantiate dataset
    dataset = MetaChexDataset(multiclass=True, baseline=True, batch_size=32, max_num_vis_samples=100)
    
    # Grab training dataset
    train_ds = dataset.train_ds

    # Load CheXNet
    chexnet = load_chexnet(dataset.num_classes_multiclass, multiclass=True, embedding_dim=128, baseline=True)
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
        eval_path = args.ckpt_save_path if args.pretrained is None else args.pretrained
        y_test_true = dataset.test_ds.get_y_true() 
        y_test_pred = chexnet.predict(dataset.test_ds, verbose=1)
        
        dir_path = os.path.dirname(eval_path)
        mean_auroc(y_test_true, y_test_pred, dataset, eval=True, dir_path=dir_path)
        mean_f1(y_test_true, y_test_pred, dataset, eval=True, dir_path=dir_path)
        average_precision(y_test_true, y_test_pred, dataset, dir_path=dir_path)

    # Generate tSNE
    if args.tsne:
        print("[INFO] Generating tSNE plots")
        chexnet_embedder = get_embedding_model(chexnet)
        tsne_datasets = [dataset.tsne1_ds, dataset.tsne2_ds]

        embedding_save_paths = [os.path.join(record_dir, 'embeddings1.pkl'), os.path.join(record_dir, 'embeddings2.pkl')]
        tsne_save_paths = [os.path.join(record_dir, 'tsne1.png'), os.path.join(record_dir, 'tsne2.png')]
        # generating embeddings can take some time. Load if possible
        
        for i in range(2):
            if os.path.exists(embedding_save_paths[i]):
                print(f"[INFO] Embeddings {i + 1} already processed. Loading from {embedding_save_paths[i]}")
                with open(embedding_save_paths[i], 'rb') as file:
                    embeddings = pickle.load(file)
            else:
                print(f"[INFO] Embeddings {i + 1} processing. Saving to {embedding_save_paths[i]}")
                embeddings = chexnet_embedder.predict(tsne_datasets[i], verbose=1)
                with open(embedding_save_paths[i], 'wb') as file:
                    pickle.dump(embeddings, file, protocol=pickle.HIGHEST_PROTOCOL)

            tsne_labels = tsne_datasets[i].get_y_true()
            print(tsne_labels.shape)
            tsne_feats = process_tSNE(embeddings, perplexity=30, n_iter=10000, learning_rate=300,
                                     early_exaggeration=12)

            plot_tsne(tsne_feats, tsne_labels, save_path=tsne_save_paths[i])
