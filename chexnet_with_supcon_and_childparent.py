import numpy as np
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from metachex.configs.config import *
from metachex.loss import Losses
from metachex.dataloader import MetaChexDataset
from metachex.utils import *
from metachex.nearest_neighbour import NearestNeighbour
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Baseline MetaChex: Fine-Tuned ChexNet')
    parser.add_argument('-t', '--tsne', action='store_true', help='Generate tSNE plot')
    parser.add_argument('-e', '--evaluate', action='store_true', help='Evaluate model performance')
    parser.add_argument('-c', '--ckpt_save_path', default='training_progress_supcon_childparent/cp_best.ckpt')
    parser.add_argument('-p', '--pretrained', default=None, help='Path to pretrained weights, if desired')
    parser.add_argument('-n', '--num_epochs', type=int, default=15, help='Number of epochs to train for')
    parser.add_argument('-wp', '--parent_weight', type=int, default=0.5, help='Weight for modifying existing parent embedding')
    parser.add_argument('-wc', '--child_weight', type=int, default=0.2, help='Weight for modifying non-existent parent embeddings')
    parser.add_argument('-w2', '--stage2_weight', type=int, default=1., help='Weight for childParent regularizer')
    return parser.parse_args()


def compile_stage(stage_num=1, parent_weight=0.5, child_weight=0.2, stage2_weight=1.):
    if stage_num == 1:
        loss_fn = Losses(embed_dim=chexnet_encoder.get_layer('embedding').output_shape[-1], batch_size=dataset.batch_size,
                        stage_num=1)
    else:
        loss_fn = Losses(child_to_parent_map=dataset.child_to_parent_map, 
                         embed_dim=chexnet_encoder.get_layer('embedding').output_shape[-1], batch_size=dataset.batch_size,
                         stage_num=2, parent_weight=parent_weight, child_weight=child_weight, stage2_weight=stage2_weight)

    chexnet_encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                    loss=loss_fn.supcon_full_loss(),
                    run_eagerly=True)
                  

def train_stage(num_epochs=15, stage_num=1, checkpoint_dir="training_progress_supcon_childparent"):
    # Create a callback that saves the model's weights
    ds = dataset.train_ds
    if stage_num == 1:
        checkpoint_path = os.path.join(checkpoint_dir, "stage1_cp_best.ckpt")
#         ds = dataset.stage1_ds
    else:
        checkpoint_path = os.path.join(checkpoint_dir, "stage2_cp_best.ckpt")
#         ds = dataset.train_ds
        
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)
    
    hist = chexnet_encoder.fit(ds,
        epochs=num_epochs,
        steps_per_epoch=dataset.train_steps_per_epoch, 
        batch_size=dataset.batch_size, 
        callbacks=[cp_callback]
        )

    with open(os.path.join(checkpoint_dir, 'trainHistoryDict'), 'wb') as file_pi:
            pickle.dump(hist.history, file_pi)

    return hist     
        
     
def create_parent_embed_matrix(model, dataset, max_num_samples_per_class=20, parent_emb_path="training_progress/parent_emb.npy"):
    
    nn = NearestNeighbour(model=model, dataset=dataset, parents_only=True)
    nn.calculate_prototypes(full=False, max_per_class=max_num_samples_per_class)
    
    embed_matrix = nn.prototypes.T ## (embedding_dim, 27)
    
    np.save(parent_emb_path, embed_matrix)
    
    return embed_matrix
    
        
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
    
    print(chexnet_encoder.summary())
          
    # Compile stage 1
    compile_stage(stage_num=1)
    
    checkpoint_dir="training_progress_supcon_childparent"
    # Get weights
    if args.pretrained is None:
        print("[INFO] Beginning Fine Tuning")
        # Train stage 1
        stage1_hist = train_stage(num_epochs=args.num_epochs, stage_num=1)
        
        # Create embedding matrix for parents
        embed_matrix = create_parent_embed_matrix(model=chexnet_encoder, dataset=dataset, max_num_samples_per_class=2) # change #
        
        # Compile stage 2
        compile_stage(stage_num=2, parent_weight=args.parent_weight, child_weight=args.child_weight,
                      stage2_weight=args.stage2_weight)
        stage2_hist = train_stage(num_epochs=args.num_epochs, stage_num=2)
        
        record_dir = checkpoint_dir
    else:
        print("[INFO] Loading weights")
        # Load weights
        chexnet_encoder.load_weights(args.pretrained)
        record_dir = os.path.dirname(args.pretrained)
            
    # Evaluate
    nn = NearestNeighbour(model=chexnet_encoder, dataset=dataset)
    nn.calculate_prototypes(full=False, max_per_class=2) ## realistically, change to larger number (20)
    
    if args.evaluate:
        print("[INFO] Evaluating performance")
        y_test_labels = dataset.test_ds.get_y_true()
        y_test_embeddings = chexnet_encoder.predict(dataset.test_ds, verbose=1)
        y_pred = nn.get_nearest_neighbour(y_test_embeddings)
        
        ## metrics
        auroc = mean_auroc(y_test_labels, y_pred, dataset, eval=True, dir_path=checkpoint_dir)
        mean_AP = average_precision(y_test_labels, y_pred, dataset, dir_path=checkpoint_dir)
        

    # Generate tSNE
    if args.tsne:
        print("[INFO] Generating tSNE plots")
        tsne_dataset = MetaChexDataset(shuffle_train=False)

        embedding_save_path = os.path.join(record_dir, 'embeddings_supcon.npy')
        # generating embeddings can take some time. Load if possible
        if os.path.isfile(embedding_save_path):
            print(f"[INFO] Embeddings already processed. Loading from {embedding_save_path}")
            training_embeddings = np.load(embedding_save_path)
        else:
            print(f"[INFO] Embeddings processing. Saving to {embedding_save_path}")
            training_embeddings = chexnet_encoder.predict(tsne_dataset.train_ds, verbose=1)
            np.save(embedding_save_path, training_embeddings)

        tsne_feats = process_tSNE(training_embeddings)
        tsne_labels = tsne_dataset.train_ds.get_y_true()

        plot_tsne(tsne_feats, tsne_labels, lables_names=tsne_dataset.unique_labels)

    
    
                  
                  

