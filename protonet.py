import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse
# import wandb
# from wandb.keras import WandbCallback

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
    parser.add_argument('-c', '--ckpt_save_path', default='training_progress_protonet/cp_best.ckpt')
    parser.add_argument('-p', '--pretrained', default=None, help='Path to pretrained weights, if desired')
    parser.add_argument('-n', '--num_epochs', type=int, default=150, help='Number of epochs to train for')
    return parser.parse_args()


def train(num_epochs=15, checkpoint_path="training_progress_protonet/cp_best.ckpt"):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    monitor='val_proto_acc',
                                                    mode='max',
                                                    save_best_only=True)

    hist = chexnet_encoder.fit(dataset.train_ds,
                validation_data=dataset.val_ds,
                epochs=num_epochs,
                steps_per_epoch=dataset.num_meta_train_episodes, 
                validation_steps=100, 
                callbacks=[cp_callback] #, WandbCallback()]
                )

    with open(os.path.join(checkpoint_dir, 'trainHistoryDict'), 'wb') as file_pi:
            pickle.dump(hist.history, file_pi)

    return hist


def compile():
    loss_fn = Losses(num_classes=dataset.n, num_samples_per_class=dataset.k, num_query=dataset.n_query)

    chexnet_encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                           loss=loss_fn.proto_loss(),
                           metrics=[proto_acc_outer(num_classes=dataset.n, 
                                              num_samples_per_class=dataset.k, 
                                              num_query=dataset.n_query),
                                    proto_sup_acc_outer(num_classes=dataset.n, 
                                              num_samples_per_class=dataset.k, 
                                              num_sup=dataset.n*dataset.k), 
                                    proto_mean_auroc_outer(num_classes=dataset.n, 
                                              num_samples_per_class=dataset.k, 
                                              num_query=dataset.n_query),
                                    proto_mean_f1_outer(num_classes=dataset.n, 
                                              num_samples_per_class=dataset.k, 
                                              num_query=dataset.n_query)],
                           run_eagerly=True)


def eval():
    loss_fn = Losses(num_classes=dataset.n_test, num_samples_per_class=dataset.k_test, num_query=dataset.n_test_query)
    
    chexnet_encoder.compile(loss=loss_fn.proto_loss(),
                            metrics=[proto_acc_outer(num_classes=dataset.n_test, 
                                              num_samples_per_class=dataset.k_test, 
                                              num_query=dataset.n_test_query), 
                                     proto_acc_covid_outer(num_classes=dataset.n_test, 
                                              num_samples_per_class=dataset.k_test, 
                                              num_query=dataset.n_test_query), 
                                     proto_acc_tb_outer(num_classes=dataset.n_test, 
                                              num_samples_per_class=dataset.k_test, 
                                              num_query=dataset.n_test_query), 
                                     proto_acc_covid_tb_outer(num_classes=dataset.n_test, 
                                              num_samples_per_class=dataset.k_test, 
                                              num_query=dataset.n_test_query), 
                                     proto_mean_auroc_outer(num_classes=dataset.n_test, 
                                              num_samples_per_class=dataset.k_test, 
                                              num_query=dataset.n_test_query),
                                     proto_mean_auroc_covid_outer(num_classes=dataset.n_test, 
                                              num_samples_per_class=dataset.k_test, 
                                              num_query=dataset.n_test_query),
                                     proto_mean_auroc_tb_outer(num_classes=dataset.n_test, 
                                              num_samples_per_class=dataset.k_test, 
                                              num_query=dataset.n_test_query),
#                                      proto_mean_auroc_covid_tb_outer(num_classes=dataset.n_test, 
#                                               num_samples_per_class=dataset.k_test, 
#                                               num_query=dataset.n_test_query),
                                     proto_mean_f1_outer(num_classes=dataset.n_test, 
                                              num_samples_per_class=dataset.k_test, 
                                              num_query=dataset.n_test_query),
                                     proto_mean_f1_covid_outer(num_classes=dataset.n_test, 
                                              num_samples_per_class=dataset.k_test, 
                                              num_query=dataset.n_test_query),
                                     proto_mean_f1_tb_outer(num_classes=dataset.n_test, 
                                              num_samples_per_class=dataset.k_test, 
                                              num_query=dataset.n_test_query),
#                                      proto_mean_f1_covid_tb_outer(num_classes=dataset.n_test, 
#                                               num_samples_per_class=dataset.k_test, 
#                                               num_query=dataset.n_test_query)
                                    ],
                            run_eagerly=True) 

    chexnet_encoder.evaluate(dataset.test_ds, steps=dataset.num_meta_test_episodes)


def load_model():
    chexnet_encoder = load_chexnet(1, embedding_dim=64) ## any number will do, since we get rid of final dense layer
    chexnet_encoder = get_embedding_model(chexnet_encoder)
    chexnet_encoder.trainable = True
    
    return chexnet_encoder


if __name__ == '__main__':
    args = parse_args()
#     wandb.init(project="protonet-baby-1", entity="edwinpan")

    # os.environ["CUDA_VISIBLE_DEVICES"]=""
    tf.test.is_gpu_available()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Instantiate dataset
    dataset = MetaChexDataset(protonet=True, batch_size=1, n=3, k=10, n_query=5, 
                              n_test=3, k_test=10, n_test_query=5, 
                              num_meta_train_episodes=100, num_meta_val_episodes=20, num_meta_test_episodes=10,
                              max_num_vis_samples=100)
    
    # Load CheXNet
    chexnet_encoder = load_model()
    
#     print(chexnet_encoder.summary())
    
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
        chexnet_encoder.load_weights(args.pretrained)
        record_dir = os.path.dirname(args.pretrained)

    # Evaluate
    if args.evaluate:
        print("[INFO] Evaluating performance")
        eval() ## protoloss, proto_acc, proto_mean_auroc
        
    # Generate tSNE
    if args.tsne:
        print("[INFO] Generating tSNE plots")
        chexnet_embedder = chexnet_encoder
        tsne_datasets = [dataset.tsne1_ds, dataset.tsne2_ds]

        embedding_save_paths = [os.path.join(record_dir, 'embeddings1.pkl'), os.path.join(record_dir, 'embeddings2.pkl')]
        tsne_save_paths = [os.path.join(record_dir, 'covid_pneumonia_child_protonet.pdf'), 
                           os.path.join(record_dir, 'hernia_lung_opacity_pneumonia_protonet.pdf')]
        # generating embeddings can take some time. Load if possible
        
        ## TSNE 1: Parents and their children
        learning_rates = [500, 300]
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
            tsne_feats = process_tSNE(embeddings, perplexity=30, n_iter=10000, learning_rate=learning_rates[i],
                                     early_exaggeration=12)

            plot_tsne(tsne_feats, tsne_labels, save_path=tsne_save_paths[i])
