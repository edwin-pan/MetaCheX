import numpy as np
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from metachex.configs.config import *
from metachex.loss import Losses
from metachex.dataloader import MetaChexDataset, ImageSequence
from metachex.utils import *
from sklearn.metrics.pairwise import euclidean_distances
from chexnet_with_supcon import NearestNeighbour


def compile_stage(stage_num=1):
    if stage_num == 1:
        loss_fn = Losses(child_to_parent_map=dataset.child_to_parent_map, num_indiv_parents=dataset.num_classes_multitask,
                        embed_dim=chexnet_encoder.get_layer('embedding').output_shape[-1], batch_size=dataset.batch_size,
                        stage_num=1)
    else:
        loss_fn = Losses(child_to_parent_map=dataset.child_to_parent_map, num_indiv_parents=dataset.num_classes_multitask,
                        embed_dim=chexnet_encoder.get_layer('embedding').output_shape[-1], batch_size=dataset.batch_size,
                        stage_num=2)

    chexnet_encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                    loss=loss_fn.supcon_full_loss(),
                    run_eagerly=True)
                  

def train_stage(num_epochs=15, stage_num=1, checkpoint_dir="training_progress_supcon_childparent"):
    # Create a callback that saves the model's weights
    if stage_num == 1:
        checkpoint_path = os.path.join(checkpoint_dir, "stage1_cp_best.ckpt")
        ds = dataset.stage1_ds
        
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)
    
    hist = chexnet_encoder.fit(ds,
        epochs=num_epochs,
        steps_per_epoch=dataset.stage1_steps_per_epoch, 
        batch_size=dataset.batch_size, 
        callbacks=[cp_callback]
        )

#     with open(os.path.join(checkpoint_dir, 'trainHistoryDict'), 'wb') as file_pi:
#             pickle.dump(hist.history, file_pi)

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
    
    print(chexnet_encoder.summary())
          
    # Compile stage 1
    compile_stage(stage_num=1)
    
    checkpoint_dir="training_progress_supcon_childparent"
    # Get weights
    if args.pretrained is None:
        print("[INFO] Beginning Fine Tuning")
        # Train stage 1
        stage1_hist = train_stage(num_epochs=args.num_epochs, stage_num=1)
        
        # Compile stage 2
        compile_stage(stage_num=2)
        stage2_hist = train_stage(num_epochs=args.num_epochs, stage_num=2)
        
        record_dir = checkpoint_dir
    else:
        print("[INFO] Loading weights")
        # Load weights
        chexnet_encoder.load_weights(args.pretrained)
        record_dir = os.path.dirname(args.pretrained)
            
    # Evaluate
    nn = NearestNeighbour(model=chexnet_encoder, num_classes=dataset.num_classes_multiclass)
    nn.calculate_prototypes(dataset.train_ds, full=False, max_per_class=2) ## realistically, change to larger number (20)
    
    if args.evaluate:
        print("[INFO] Evaluating performance")
        y_test_labels = dataset.test_ds.get_y_true()
        y_test_embeddings = chexnet_encoder.predict(dataset.test_ds, verbose=1)
        y_pred = nn.get_nearest_neighbour(y_test_embeddings)
        
        ## CONTINUE
        auroc = mean_auroc(y_test_labels, y_pred, dataset, eval=True)
        mean_AP = average_precision(y_test_labels, y_pred, dataset)
        

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

    
    
                  
                  

