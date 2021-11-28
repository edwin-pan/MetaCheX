import numpy as np
import pandas as pd
import tensorflow as tf
from metachex.image_sequence import ImageSequence
from sklearn.metrics.pairwise import euclidean_distances
from metachex.utils import get_sampled_df_multiclass

class NearestNeighbour():
    
    def __init__(self, model, dataset, parents_only=False):
        """
        If parents_only is True, num_classes == 27
        """
        if parents_only:
            self.num_classes = dataset.num_classes_multitask
        else:
            self.num_classes = dataset.num_classes_multiclass
        
        embedding_dim = model.get_layer('embedding').output_shape[-1]
        self.prototypes = np.zeros((embedding_dim, self.num_classes))
        self.model = model
        self.dataset = dataset
        self.parents_only = parents_only
    
    
    def load_prototypes(self, dir_path="."):
        
        save_path = os.path.join(dir_path, "prototypes.npy")
        if os.path.exists(save_path):
            self.prototypes = np.load(save_path)
            return self.prototypes
        else:
            raise ValueError(f'{save_path} does not exist')
    
    
    def calculate_prototypes(self, full=False, max_per_class=2, dir_path="."):
        """
        Note: this takes a long time if run full ds -- we can also sample max_per_class images per class
        """
        
        if full:
            df = self.dataset.train_ds.df
        else:
            df = get_sampled_df_multiclass(self.dataset.train_ds.df, self.num_classes, self.parents_only, max_per_class)
        
        train_ds = ImageSequence(df, shuffle_on_epoch_end=False, num_classes=self.num_classes, multiclass=True,
                                 parents_only=self.parents_only)
        
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
        
        ## Save prototypes
        save_path = os.path.join(dir_path, "prototypes.npy")
        np.save(save_path, self.prototypes)
        
        print(self.prototypes.shape)
                                 
    
    def get_nearest_neighbour(self, queries):
        """
        queries: [batch_size, embedding_dim]
        
        return:
        one-hot preds: [batch_size, num_prototypes]
        """
        
        distances = euclidean_distances(queries, self.prototypes.T)
        pred = np.argmin(distances, axis=1)
        
        return np.eye(self.prototypes.shape[1])[pred] ## one-hot
    
    
    def get_soft_predictions(self, queries):
        """
        distances: [batch_size, num_classes]
        """
        distances = euclidean_distances(queries, self.prototypes.T)
    
        soft_pred = tf.nn.softmax(logits=-1 * distances)
        
        return soft_pred
      