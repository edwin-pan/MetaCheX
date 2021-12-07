import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse
import wandb
from wandb.keras import WandbCallback

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
    parser.add_argument('-c', '--ckpt_save_path', default='training_progress_protonet_supcon/cp_best.ckpt')
    parser.add_argument('-p', '--pretrained', default=None, help='Path to pretrained weights, if desired')
    parser.add_argument('-n1', '--num_epochs_stage_1', type=int, default=30, help='Number of epochs to train stage 1 for')
    parser.add_argument('-n2', '--num_epochs_stage_2', type=int, default=56, help='Number of epochs to train stage 2 for')
    parser.add_argument('-t2', '--train_stage2', action='store_true', help='Whether to train stage 2 (after loading weights)')
    return parser.parse_args()

def compile_stage(stage_num=1):
    if stage_num == 1:
        loss_fn = Losses(embed_dim=chexnet_encoder.get_layer('embedding').output_shape[-1], batch_size=dataset.batch_size,
                        num_classes=dataset.n, num_samples_per_class=dataset.k, num_query=dataset.n_query)

        chexnet_encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                                loss=loss_fn.supcon_label_loss(proto=True),
                                loss_weights=100,
                                run_eagerly=True)
    else:
        loss_fn = Losses(num_classes=dataset.n, num_samples_per_class=dataset.k, num_query=dataset.n_query)

        chexnet_encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                                loss=loss_fn.proto_loss(),
                                metrics=[proto_acc_outer(num_classes=dataset.n, 
                                              num_samples_per_class=dataset.k, 
                                              num_query=dataset.n_query), 
                                     proto_mean_auroc_outer(num_classes=dataset.n, 
                                              num_samples_per_class=dataset.k, 
                                              num_query=dataset.n_query),
                                    proto_mean_f1_outer(num_classes=dataset.n, 
                                              num_samples_per_class=dataset.k, 
                                              num_query=dataset.n_query)],
                               run_eagerly=True)
                  

def train_stage(num_epochs=15, stage_num=1, checkpoint_dir="training_progress_protonet_supcon"):
    # Create a callback that saves the model's weights
    if stage_num == 1:
        checkpoint_path = os.path.join(checkpoint_dir, "stage1_cp_best_{epoch:02d}.ckpt")
        hist_dict_name = 'trainStage1HistoryDict'
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1,
                                                        monitor='val_loss',
                                                        mode='min',
                                                        save_best_only=False
                                                        )
        hist = chexnet_encoder.fit(dataset.train_ds,
            epochs=num_epochs,
            validation_data=dataset.val_ds,                       
            callbacks=[cp_callback]
            )
    else:
        wandb.init(project="protonet-baby-1", entity="edwinpan")
        checkpoint_path = os.path.join(checkpoint_dir, "stage2_cp_best.ckpt")
        hist_dict_name = 'trainStage2HistoryDict'
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1,
                                                        monitor='val_proto_acc',
                                                        mode='max',
                                                        save_best_only=True)


    
        hist = chexnet_encoder.fit(dataset.train_ds,
            epochs=num_epochs,
            validation_data=dataset.val_ds,                       
            callbacks=[cp_callback, WandbCallback()]
            )

    with open(os.path.join(checkpoint_dir, hist_dict_name), 'wb') as file_pi:
            pickle.dump(hist.history, file_pi)

    return hist   


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
    # os.environ["CUDA_VISIBLE_DEVICES"]=""
    tf.test.is_gpu_available()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Instantiate dataset
    dataset = MetaChexDataset(multiclass=True, protonet=True, batch_size=1, n=3, k=10, n_query=5, 
                              n_test=3, k_test=10, n_test_query=5,
                              num_meta_train_episodes=100, num_meta_val_episodes=20, num_meta_test_episodes=1000,
                              )

    # Load CheXNet
    chexnet_encoder = load_model()
    
    print(chexnet_encoder.summary())
    
    # Compile
    compile_stage(stage_num=1)

    # Get weights
    if args.pretrained is None:
        print("[INFO] Beginning Fine Tuning")
        
        # Train stage 1
        print("[INFO] Stage 1 Training")
        stage1_hist = train_stage(num_epochs=args.num_epochs_stage_1, stage_num=1)
        
        # Compile stage 2
        print("[INFO] Stage 2 Training")
        compile_stage(stage_num=2)
        stage2_hist = train_stage(num_epochs=args.num_epochs_stage_2, stage_num=2)
        
        record_dir = os.path.dirname(args.ckpt_save_path)
    else:
        print("[INFO] Loading weights")
        # Load weights
        chexnet_encoder.load_weights(args.pretrained)
        
        if args.train_stage2:
            print("[INFO] Stage 2 Training")
            compile_stage(stage_num=2)
            stage2_hist = train_stage(num_epochs=args.num_epochs_stage_2, stage_num=2)
        
        record_dir = os.path.dirname(args.pretrained)

    # Evaluate
    if args.evaluate:
        print("[INFO] Evaluating performance")
        eval() ## protoloss, proto_acc, proto_mean_auroc
    
    # Generate tSNE
    if args.tsne:
        print("[INFO] Generating tSNE plots")

        embedding_save_path = os.path.join(record_dir, 'embeddings.npy')
        sampled_ds_save_path = os.path.join(record_dir, 'sampled_ds.pkl')
        # generating embeddings can take some time. Load if possible

        if os.path.isfile(sampled_ds_save_path):
            print(f"[INFO] Loading sampled dataset {sampled_ds_save_path}")
            with open(sampled_ds_save_path, 'rb') as file:
                sampled_ds = pickle.load(file)
        else:
            print(f"[INFO] Train ds sampling. Saving to {sampled_ds_save_path}")
            sampled_ds = get_sampled_ds(eval_dataset.train_ds, multiclass=False, max_per_class=20)
            with open(sampled_ds_save_path, 'wb') as file:
                pickle.dump(sampled_ds, file)
                
        if os.path.isfile(embedding_save_path):
            print(f"[INFO] Embeddings already processed. Loading from {embedding_save_path}")
            training_embeddings = np.load(embedding_save_path)
        else:
            print(f"[INFO] Embeddings processing. Saving to {embedding_save_path}")
            training_embeddings = chexnet_embedder.predict(sampled_ds, verbose=1)
            np.save(embedding_save_path, training_embeddings)
        
        tsne_feats = process_tSNE(training_embeddings)
        tsne_labels = sampled_ds.get_y_true()

        plot_tsne(tsne_feats, tsne_labels, label_names=eval_dataset.unique_labels)
