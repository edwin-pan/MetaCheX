import numpy as np
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from metachex.configs.config import *
from metachex.loss import Losses
from metachex.dataloader import MetaChexDataset
from metachex.utils import *



def compile():
    loss_fn = Losses()

    chexnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                    loss=loss_fn.supcon_label_loss(),
                    metrics=[mean_auroc], 
                    run_eagerly=True)
                  

def train(num_epochs=15, checkpoint_path="training_progress_supcon/cp_best.ckpt"):
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
    dataset = MetaChexDataset(multiclass=True)

    # Load CheXNet
    chexnet = load_chexnet(1) ## any number will do, since we get rid of final dense layer
    chexnet = get_embedding_model(chexnet)
    chexnet.trainable = True
    
    print(chexnet.summary())
    
    # Compile
    compile()
    
    checkpoint_path="training_progress_supcon/cp_best.ckpt"
    # Get weights
    if args.pretrained is None:
        print("[INFO] Beginning Fine Tuning")
        # Train
        hist = train(args.num_epochs)
        record_dir = os.path.dirname(checkpoint_path)
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

        embedding_save_path = os.path.join(record_dir, 'embeddings_supcon.npy')
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

    
    
                  
                  

