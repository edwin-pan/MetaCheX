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
    parser.add_argument('-c', '--ckpt_save_path', default='training_progress/cp_best.ckpt')
    parser.add_argument('-p', '--pretrained', default=None, help='Path to pretrained weights, if desired')
    parser.add_argument('-n', '--num_epochs', type=int, default=15, help='Number of epochs to train for')
    return parser.parse_args()

def compile_stage(stage_num=1):
    if stage_num == 1:
        loss_fn = Losses(embed_dim=chexnet_encoder.get_layer('embedding').output_shape[-1], batch_size=dataset.batch_size)

        chexnet_encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                                loss=loss_fn.supcon_label_loss(),
                                run_eagerly=True)
    else:
        loss_fn = Losses(num_classes=dataset.n, num_samples_per_class=dataset.k, num_query=dataset.n_query)

        chexnet_encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                                loss=loss_fn.proto_loss(),
                                metrics=[proto_acc],
                                run_eagerly=True)
                  

def train_stage(num_epochs=15, stage_num=1, checkpoint_dir="training_progress_supcon_childparent"):
    # Create a callback that saves the model's weights
    ds = dataset.train_ds
    if stage_num == 1:
        checkpoint_path = os.path.join(checkpoint_dir, "stage1_cp_best.ckpt")
        hist_dict_name = 'trainStage1HistoryDict'
    else:
        checkpoint_path = os.path.join(checkpoint_dir, "stage2_cp_best.ckpt")
        hist_dict_name = 'trainStage2HistoryDict'
        
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)
    
    hist = chexnet_encoder.fit(ds,
        epochs=num_epochs,
        steps_per_epoch=dataset.train_steps_per_epoch, 
        batch_size=dataset.batch_size, 
        callbacks=[cp_callback]
        )

    with open(os.path.join(checkpoint_dir, hist_dict_name), 'wb') as file_pi:
            pickle.dump(hist.history, file_pi)

    return hist   


def eval():
    loss_fn = Losses(num_classes=dataset.n_test, num_samples_per_class=dataset.k_test, num_query=dataset.n_query_test)
    chexnet_encoder.compile(loss=loss_fn.proto_loss(),
                            metrics=[proto_acc, proto_mean_auroc],
                            run_eagerly=True)

    chexnet_encoder.evaluate(dataset.test_ds, steps=dataset.num_meta_test_episodes)


def load_model():
    chexnet_encoder = load_chexnet(1) ## any number will do, since we get rid of final dense layer
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
    dataset = MetaChexDataset(protonet=True, batch_size=1, n=3, k=5, n_query=5, 
                              n_test=3, k_test=5, n_query_test=5)

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
        
        record_dir = checkpoint_dir
    else:
        print("[INFO] Loading weights")
        # Load weights
        chexnet_encoder.load_weights(args.pretrained)
        record_dir = os.path.dirname(args.pretrained)

    # Evaluate
    if args.evaluate:
        print("[INFO] Evaluating performance")
        eval() ## protoloss, proto_acc, proto_mean_auroc
        
#         y_test_true = dataset.test_ds.get_y_true() 
#         y_test_embeddings = chexnet_encoder.predict(dataset.test_ds, verbose=1)
#         y_pred = nn.get_soft_predictions(y_test_embeddings)
        
#         dir_path = os.path.dirname(args.ckpt_save_path)
#         mean_auroc(y_test_true, y_test_pred, dataset, eval=True, dir_path=dir_path)
#         average_precision(y_test_true, y_test_pred, dataset, dir_path=dir_path)
    
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
            sampled_ds = get_sampled_ds(dataset.train_ds, multiclass=False, max_per_class=20)
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

        plot_tsne(tsne_feats, tsne_labels, label_names=dataset.unique_labels)