import numpy as np
import pandas as pd
import tensorflow as tf
from metachex.image_sequence import ImageSequence
from sklearn.metrics.pairwise import euclidean_distances

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
    
    
    def calculate_prototypes(self, full=False, max_per_class=2):
        """
        Note: this takes a long time if run full ds -- we can also sample max_per_class images per class
        """
        
        if full:
            df = self.dataset.train_ds.df
        else:
            df = self.get_sampled_df(self.dataset.train_ds.df, max_per_class)
        
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
    

    def get_sampled_df(self, train_df, max_per_class=20):
        """
        Sample max_per_class samples from each class in train_df
        if self.parents_only, sample only the parents that exist and the children of parents that don't
        """
        sampled_df = pd.DataFrame(columns=train_df.columns)

        if not self.parents_only:
            for i in range(self.num_classes):
                df_class = train_df[train_df['label_num_multi'] == i]

                if len(df_class) > max_per_class:
                    df_class = df_class.sample(max_per_class)

                sampled_df = sampled_df.append(df_class)
                
        else: ## to get parent embedding matrix
            label_multitask_arr = np.array(train_df['label_multitask'].to_list()) ## [len(train_df), 27]
            row_indices, multitask_indices = np.where(label_multitask_arr == 1)
            
            for i, label in enumerate(self.dataset.parent_multiclass_labels):
                if label != -1: ## Sample parents that exist individually
                    df_class = train_df[train_df['label_num_multi'] == label]

                else: ## Sample children of parents that don't exist individually
                    ## Get rows where multitask_indices includes i
                    children_rows = row_indices[multitask_indices == i]
                    df_class = train_df.iloc[children_rows]
                    
                if len(df_class) > max_per_class:
                    df_class = df_class.sample(max_per_class)
                
                df_class['parent_id'] = i ## label with parent class
                sampled_df = sampled_df.append(df_class)
           
        sampled_df = sampled_df.reset_index(drop=True)
          
        return sampled_df