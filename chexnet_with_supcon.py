import numpy as np
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from metachex.configs.config import *
from metachex.loss import Losses
from metachex.dataloader import MetaChexDataset
from metachex.utils import *
import argparse
from metachex.nearest_neighbour import NearestNeighbour

def parse_args():
    parser = argparse.ArgumentParser(description='Baseline MetaChex: Fine-Tuned ChexNet')
    parser.add_argument('-t', '--tsne', action='store_true', help='Generate tSNE plot')
    parser.add_argument('-e', '--evaluate', action='store_true', help='Evaluate model performance')
    parser.add_argument('-c', '--ckpt_save_path', default='training_progress_supcon/cp_best.ckpt')
    parser.add_argument('-p', '--pretrained', default=None, help='Path to pretrained weights, if desired')
    parser.add_argument('-n', '--num_epochs', type=int, default=15, help='Number of epochs to train for')
    return parser.parse_args()
    

def compile():
    loss_fn = Losses(embed_dim=chexnet_encoder.get_layer('embedding').output_shape[-1], batch_size=dataset.batch_size)

    chexnet_encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                    loss=loss_fn.supcon_label_loss(),
                    # metrics=
                    run_eagerly=True)
                  

def train_model(num_epochs=15, checkpoint_path="training_progress_supcon/cp_best.ckpt"):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

    hist = chexnet_encoder.fit(dataset.train_ds,
                epochs=num_epochs,
                steps_per_epoch=dataset.train_steps_per_epoch, ## size(train_ds) * 0.125 * 0.1
                batch_size=dataset.batch_size, 
                callbacks=[cp_callback]
                )

    with open(os.path.join(checkpoint_dir, 'trainHistoryDict'), 'wb') as file_pi:
            pickle.dump(hist.history, file_pi)

    return hist     
        
        
if __name__ == '__main__':
    args = parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"]=""
    tf.test.is_gpu_available()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Instantiate dataset
    dataset = MetaChexDataset(multiclass=True, batch_size=32)

    # Load CheXNet
    chexnet_encoder = load_chexnet(1) ## any number will do, since we get rid of final dense layer
            
    chexnet_encoder = get_embedding_model(chexnet_encoder)
    chexnet_encoder.trainable = True
    
#     print(chexnet_encoder.summary())
            
    # Compile
    compile()
    
    checkpoint_path="training_progress_supcon/cp_best.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Get weights
    if args.pretrained is None:
        print("[INFO] Beginning Fine Tuning")
        # Train
        hist = train_model(args.num_epochs)
        record_dir = os.path.dirname(checkpoint_path)
    else:
        print("[INFO] Loading weights")
        # Load weights
        chexnet_encoder.load_weights(args.pretrained)
        record_dir = os.path.dirname(args.pretrained)
            
    # Evaluate
    nn = NearestNeighbour(model=chexnet_encoder, dataset=dataset)
    nn.calculate_prototypes(full=False, max_per_class=2) ## realistically, change to larger number (20)
    
    print(f'prototypes shape: {nn.prototypes.shape}')
#     print(nn.prototypes.T)
    
    if args.evaluate:
        print("[INFO] Evaluating performance")
        y_test_labels = dataset.test_ds.get_y_true()
        y_test_embeddings = chexnet_encoder.predict(dataset.test_ds, verbose=1)
        y_pred = nn.get_soft_predictions(y_test_embeddings)
        
        ## Metrics
        auroc = mean_auroc(y_test_labels, y_pred, dataset, eval=True, dir_path=checkpoint_dir)
        mean_AP = average_precision(y_test_labels, y_pred, dataset, dir_path=checkpoint_dir)
        

    # Generate tSNE
    if args.tsne:
        print("[INFO] Generating tSNE plots")
        tsne_dataset = MetaChexDataset(shuffle_train=False)

        embedding_save_path = os.path.join(record_dir, 'embeddings.npy')
        sampled_ds_save_path = os.path.join(record_dir, 'sampled_ds.pkl')
        # generating embeddings can take some time. Load if possible
        if os.path.isfile(embedding_save_path) and os.path.isfile(sampled_ds_save_path):
            print(f"[INFO] Embeddings already processed. Loading from {embedding_save_path}")
            training_embeddings = np.load(embedding_save_path)
            
            print(f"[INFO] Loading sampled dataset {sampled_ds_save_path}")
            with open(sampled_ds_save_path, 'rb') as file:
                sampled_ds = pickle.load(file)
                
        else:
            print(f"[INFO] Train ds sampled. Saving to {sampled_ds_save_path}")
            sampled_ds = get_sampled_ds(tsne_dataset.train_ds, multiclass=True, max_per_class=20)
            with open(sampled_ds_save_path, 'wb') as file:
                pickle.dump(sampled_ds_save_path, file)
                
            print(f"[INFO] Embeddings processing. Saving to {embedding_save_path}")
            training_embeddings = chexnet_encoder.predict(sampled_ds, verbose=1)
            np.save(embedding_save_path, training_embeddings)

        tsne_feats = process_tSNE(training_embeddings)
        tsne_labels = sampled_ds.get_y_true()

        plot_tsne(tsne_feats, tsne_labels, label_names=tsne_dataset.unique_labels)