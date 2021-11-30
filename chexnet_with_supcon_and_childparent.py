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
    parser.add_argument('-n1', '--num_epochs_stage_1', type=int, default=15, help='Number of epochs to train stage 1 for')
    parser.add_argument('-n2', '--num_epochs_stage_2', type=int, default=15, help='Number of epochs to train stage 2 for')
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
    checkpoint_path = os.path.join(checkpoint_dir, f"stage{stage_num}_cp_best.ckpt")
    hist_dict_name = f'trainStage{stage_num}HistoryDict'

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
    
    # Get weights
    if args.pretrained is None:
        print("[INFO] Beginning Fine Tuning")
        
        # Train stage 1
        print("[INFO] Stage 1 Training")
        stage1_hist = train_stage(num_epochs=args.num_epochs_stage_1, stage_num=1)
        
        # Create embedding matrix for parents
        embed_matrix = create_parent_embed_matrix(model=chexnet_encoder, dataset=dataset, max_num_samples_per_class=2) # change #
        
        # Compile stage 2
        print("[INFO] Stage 2 Training")
        compile_stage(stage_num=2, parent_weight=args.parent_weight, child_weight=args.child_weight,
                      stage2_weight=args.stage2_weight)
        stage2_hist = train_stage(num_epochs=args.num_epochs_stage_2, stage_num=2)
        
        record_dir = os.path.dirname(args.ckpt_save_path)
    else:
        print("[INFO] Loading weights")
        # Load weights
        chexnet_encoder.load_weights(args.pretrained)
        record_dir = os.path.dirname(args.pretrained)
            
    # Evaluate
    print("[INFO] Calculating prototypes")
    nn = NearestNeighbour(model=chexnet_encoder, dataset=dataset)
    nn.calculate_prototypes(full=False, max_per_class=2) ## realistically, change to larger number (20)
    
    if args.evaluate:
        print("[INFO] Evaluating performance")
        y_test_labels = dataset.test_ds.get_y_true()
        
        y_test_embed_file = os.path.join(record_dir, 'y_test_embed.pkl')
        if os.path.exists(y_test_embed_file):
            print("[INFO] Loading y_test_embeddings")
            with open(y_test_embed_file, 'rb') as file:
                y_test_embeddings = pickle.load(file)
        else:
            print("[INFO] Generating y_test_embeddings")
            y_test_embeddings = chexnet_encoder.predict(dataset.test_ds, verbose=1)
            with open(y_test_embed_file, 'wb') as file:
                pickle.dump(y_test_embeddings, file)
        
        print("[INFO] Calculating soft predictions")
        y_pred = nn.get_soft_predictions(y_test_embeddings)
        
        ## metrics
        print("[INFO] Calculating metrics")
        auroc = mean_auroc(y_test_labels, y_pred, dataset, eval=True, dir_path=record_dir)
        mean_AP = average_precision(y_test_labels, y_pred, dataset, dir_path=record_dir)
        

    # Generate tSNE
    if args.tsne:
        print("[INFO] Generating tSNE plots")
        #tsne_dataset = MetaChexDataset(shuffle_train=False)

        embedding_save_path = os.path.join(record_dir, 'embeddings.npy')
        sampled_ds_save_path = os.path.join(record_dir, 'sampled_ds.pkl')
        tsne_save_path = os.path.join(record_dir, 'tsne.png')
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

        plot_tsne(tsne_feats, tsne_labels, label_names=dataset.unique_labels, save_path=tsne_save_path)

    
    
                  
                  

