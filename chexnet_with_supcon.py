import numpy as np
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from metachex.configs.config import *
from metachex.loss import Losses
from metachex.dataloader import MetaChexDataset, ImageSequence
from metachex.utils import *
from sklearn.metrics.pairwise import euclidean_distances

class NearestNeighbour():
    
    def __init__(self, model, num_classes):
        embedding_dim = model.get_layer('embedding').output_shape[-1]
        self.prototypes = np.zeros((embedding_dim, num_classes))
        self.model = model
        self.num_classes = num_classes
    
    def calculate_prototypes(self, train_ds, full=False, max_per_class=2):
        """
        train_ds: ImageSequence (batched (image, label))
        
        Note: this takes a long time if run full ds -- we can also sample max_per_class images per class
        """
        
        if full:
            df=train_ds.df
        else:
            df = get_sampled_df(train_ds.df, max_per_class)
        
        train_ds = ImageSequence(df, shuffle_on_epoch_end=False, num_classes=train_ds.num_classes, multiclass=True)
        
        embedding_sums = np.zeros_like(self.prototypes)
        counts = np.zeros((1, self.num_classes))
        
        labels = train_ds.get_y_true()
        print(labels.shape)
        embeddings = self.model.predict(train_ds, verbose=1)
        print(embeddings.shape)
        
        for i in range(self.num_classes):
            rows = np.where(labels[:, i] == 1)
            embeddings_for_label = embeddings[rows]

            # update
            counts[0, i] += embeddings_for_label.shape[0]
            embedding_sums[:, i] = np.sum(embeddings_for_label, axis=0)
        
        assert(not np.any(counts == 0))
        self.prototypes = embedding_sums / counts
        print(self.prototypes.shape)
                                 
    
    def get_nearest_neighbour(self, queries):
        """
        queries: [batch_size, embedding_dim]
        
        return:
        pred: [batch_size, num_prototypes]
        """
        
        distances = euclidean_distances(queries, tf.transpose(self.prototypes))
        pred = np.argmin(distances, axis=1)
        
        return np.eye(self.prototypes.shape[1])[pred] ## one-hot
    

def get_sampled_df(train_df, max_per_class=20):
    """
    Sample max_per_class samples from each class in train_df
    """
    sampled_df = pd.DataFrame(columns=train_df.columns)
    
    for i in range(dataset.num_classes_multiclass):
        df_class = train_df[train_df['label_num_multi'] == i]
        
        if len(df_class) > max_per_class:
            df_class = df_class.sample(max_per_class)
        
        sampled_df = sampled_df.append(df_class)
    
    sampled_df = sampled_df.reset_index(drop=True)
    return sampled_df
    

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

    
    
                  
                  

