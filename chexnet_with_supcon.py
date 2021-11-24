import numpy as np
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from metachex.configs.config import *
from metachex.loss import Losses
from metachex.dataloader import MetaChexDataset
from metachex.utils import *
from sklearn.metrics.pairwise import euclidean_distances


class NearestNeighbour():
    
    
    def __init__(self, class_names, embedding_dim):
        self.prototypes = np.zeros((len(class_names), embedding_dim))
        self.class_names = class_names
    
    def calculate_prototypes(self, embeddings):
        """
        embeddings: list of matrices of embeddings (list of length num_classes; matrix size [M, D], where M = # examples)
        """
        
        for i in range(len(embeddings)):
            self.prototypes[i] = np.mean(embeddings[i], axis=0)
                                 
    
    def get_nearest_neighbour(self, queries):
        """
        queries: [batch_size, embedding_dim]
        
        return:
        pred: [batch_size, num_prototypes]
        """
        
        distances = euclidean_distances(queries, self.prototypes)
        pred = np.argmin(distances, axis=1)
        
        return pred
    


def compile():
    loss_fn = Losses()

    chexnet_encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                    loss=loss_fn.supcon_label_loss(),
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
                batch_size=32, ## 8
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
    chexnet_encoder = load_chexnet(1) ## any number will do, since we get rid of final dense layer
    chexnet_encoder = get_embedding_model(chexnet_encoder)
    chexnet_encoder.trainable = True
    
    print(chexnet_encoder.summary())
    
    # Compile
    compile()
    
    checkpoint_path="training_progress_supcon/cp_best.ckpt"
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
    nn = NearestNeighbour(embeddings=train_embeddings, ...)
    nn.calculate_prototypes()
    
    if args.evaluate:
        print("[INFO] Evaluating performance")
        y_test_true = dataset.test_ds.get_y_true() 
        y_test_embeddings = chexnet_encoder.predict(dataset.test_ds, verbose=1)
        y_pred = nn.get_nearest_neighbour(y_test_embeddings)
        
        ## CONTINUE
        

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

    
    
                  
                  

