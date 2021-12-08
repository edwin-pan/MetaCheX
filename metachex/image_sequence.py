from tensorflow.keras.utils import Sequence
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import sklearn
from skimage.transform import resize

np.random.seed(271)

            
class ImageSequence(Sequence):
    """
    Class was adapted from CheXNet implementation
    """

    def __init__(self, df, batch_size=8,
                 target_size=(224, 224), augmenter=None, verbose=0, steps=None,
                 shuffle_on_epoch_end=True, random_state=271, num_classes=18, multiclass=False,
                 parents_only=False, tsne=False):
        """
        :param df: dataframe of all the images for a specific split (train, val or test)
        :param batch_size: int
        :param target_size: tuple(int, int)
        :param augmenter: imgaug object. Do not specify resize in augmenter.
                          It will be done automatically according to input_shape of the model.
        :param verbose: int
        """
        self.df = df
        self.batch_size = batch_size
        self.target_size = target_size
        self.verbose = verbose
        self.shuffle = shuffle_on_epoch_end
        self.random_state = random_state
        self.num_classes = num_classes
        self.multiclass = multiclass
        self.parents_only = parents_only
        self.tsne = tsne
        self.prepare_dataset()
        if steps is None:
            self.steps = int(np.ceil(len(self.x_path) / float(self.batch_size)))
        else:
            self.steps = int(steps)

    def __bool__(self):
        return True

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        
        end_idx = min((idx + 1) * self.batch_size, self.x_path.shape[0])
        
        batch_x_path = self.x_path[idx * self.batch_size:end_idx]
        batch_x = np.asarray([self.load_image(x_path) for x_path in batch_x_path])
        batch_x = self.transform_batch_images(batch_x)
        batch_y = self.y[idx * self.batch_size:end_idx]
        
        return batch_x, batch_y

    def load_image(self, image_path):
        image = Image.open(image_path)
        image_array = np.asarray(image.convert("RGB"))
        image_array = image_array / 255.
        image_array = resize(image_array, self.target_size)
        return image_array

    def transform_batch_images(self, batch_x):
#         if self.augmenter is not None:
#             batch_x = self.augmenter.augment_images(batch_x)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        batch_x = (batch_x - imagenet_mean) / imagenet_std
        return batch_x
    
    def get_y_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.
        """
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        
        if self.tsne:
            return self.y
        else:
            end_idx = min(self.steps*self.batch_size, self.y.shape[0])
            return self.y[:end_idx, :]

    def prepare_dataset(self):
        df = self.df.sample(frac=1., random_state=self.random_state)
        self.x_path = df['image_path']
        
        if not self.tsne:
            if self.parents_only: ## this is just to get parent embeddings later on
                self.y = np.eye(int(self.num_classes))[df['parent_id'].values.astype(int)]
            elif self.multiclass:
                self.y = np.eye(int(self.num_classes))[df['label_num_multi'].values.astype(int)]
            else:
                self.y = np.array(df['label_multitask'].to_list())
            print("self.y.shape: ", self.y.shape)
        else:
            self.y = df['label_str'].values

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()


class ProtoNetImageSequence(ImageSequence):
    
    def __init__(self, df, num_classes, num_samples_per_class, num_queries,
                 batch_size=1, target_size=(224, 224), verbose=0, steps=None, shuffle_on_epoch_end=True, 
                 random_state=271, eval=False, tsne=False):
        
        self.df = df
        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class
        self.num_queries = num_queries
        self.batch_size = batch_size
        self.target_size = target_size
        self.verbose = verbose
        self.shuffle = shuffle_on_epoch_end
        self.random_state = random_state
        self.steps = steps
        self.eval = eval
        self.tsne = tsne
        self.prepare_dataset()
        
        
    def get_y_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.
        """
        if self.shuffle or not self.eval:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True or 'eval' is False.
            """)
        
        support_labels = self.label_batches.reshape(self.steps, -1, 2) # (num_steps, n * k, 2)
        query_labels = self.query_label_batches # (num_steps, n_query, 2)
        query_masks = self.query_mask_batches # # (num_steps, n_query, 2)

        return np.concatenate((support_labels, query_labels, query_masks), axis=1) # (num_steps, n * k + 2 * n_query, 2)
        
        
    def __getitem__(self, idx):
        
        """
        batch_x: [num_classes x num_samples_per_class + num_queries, 224, 224, 3]
        batch_y: [num_classes x num_samples_per_class + num_queries + num_queries, 2] 
           - for batch_y[:num_class x num_samples_per_class + num_queries] -- proto_labels: batch_y[:, 0];
                                                                           -- multiclass_labels: batch_y[:, 1]
           - for batch_y[-num_queries: ] -- covid mask: batch_y[:, 0]; tb_mask: batch_y[:, 1]
        """
        batch_x_path = self.path_batches[idx].flatten()
        batch_q = np.asarray([self.load_image(x_path) for x_path in self.query_path_batches[idx]]) 
        batch_x = np.asarray([self.load_image(x_path) for x_path in batch_x_path])
        batch_x = np.concatenate((batch_x, batch_q), axis=0)
        batch_x = self.transform_batch_images(batch_x)
        
        batch_y = self.label_batches[idx].reshape(-1, 2) ## categorical
        batch_q_label = self.query_label_batches[idx] ## categorical
        batch_y = np.concatenate((batch_y, batch_q_label), axis=0)
        
        if self.eval:
            batch_q_masks = self.query_mask_batches[idx]
            batch_y = np.concatenate((batch_y, batch_q_masks), axis=0)
        
        return batch_x, batch_y
    
    def prepare_dataset(self):
        """
        Try to divide n_query among all classes as evenly as possible
        
        self.query_mask_batches: for axis=2, column 0 = covid mask; column 1: tb mask; column 2: covid|tb mask
        """
        all_path_batches, all_label_batches = [], []
        all_query_path_batches, all_query_label_batches = [], []
        all_query_mask_batches = []
        
        all_multiclass_label_batches = []
        
        for i in range(self.steps):
            ## sample num_classes from the classes in df
            all_classes = self.df['label_str'].drop_duplicates().values
            sampled_classes = np.random.choice(all_classes, self.num_classes)
            
            truncated_df = self.df[self.df['label_str'].isin(sampled_classes)]
            
            ## for each of the num_classes, sample num_samples_per_class samples per class
            sampled_query_df = pd.DataFrame(columns=['label_str', 'image_path', 'label_num_multi']) 
            extra_query_df = pd.DataFrame(columns=['label_str', 'image_path', 'label_num_multi']) 
            
            path_batch = []
            label_batch = []
            multiclass_label_batch = []
            total_queries = 0
            for j, classname in enumerate(sampled_classes):
                df_class = truncated_df[truncated_df['label_str'] == classname]
                df_class_sampled = df_class.sample(n=self.num_samples_per_class)
                
                df_class_query = pd.concat([df_class[['label_str', 'image_path', 'label_num_multi']], 
                                            df_class_sampled[['label_str', 'image_path', 'label_num_multi']]]
                                          ).drop_duplicates(keep=False) 
                df_class_query['label'] = j
                
                ## Sample queries
                query_sample_num = min(self.num_queries - total_queries, 
                                       min(len(df_class_query), 
                                           max(1, self.num_queries // self.num_classes)))
                total_queries += query_sample_num
                sampled_query_df = sampled_query_df.append(df_class_query.sample(n=query_sample_num))
                
                ## All the extra possible queries that were not sampled
                extra_query_df = pd.concat([df_class_query[['label_str', 'image_path', 'label_num_multi', 'label']],
                                            sampled_query_df[['label_str', 'image_path', 'label_num_multi', 'label']]]
                                          ).drop_duplicates(keep=False)
                
                path_batch.append(df_class_sampled['image_path'].values)
                
                ## Get multiclass label for sampled class
                multiclass_label = df_class['label_num_multi'].drop_duplicates().values[0]
                label_batch.append([[j, multiclass_label]] * self.num_samples_per_class)
            
            
            other_sampled_query_df = extra_query_df.sample(n=max(0, self.num_queries-total_queries))
            sampled_query_df = sampled_query_df.append(other_sampled_query_df).reset_index(drop=True)
            
            query_path_batch = sampled_query_df['image_path'].to_numpy()
            query_label_batch = sampled_query_df['label'].to_numpy().astype(int) ## categorical
            query_multiclass_label_batch = sampled_query_df['label_num_multi'].to_numpy().astype(int) ## multiclass labels

            query_label_batch = np.stack([query_label_batch, query_multiclass_label_batch], axis=1)
            
            # COVID-19/TB query masks
            if self.eval:
                query_covid_mask = sampled_query_df['label_str'].values == 'COVID-19'
                query_tb_mask = sampled_query_df['label_str'].values == 'Tuberculosis'
#                 query_covid_tb_mask = query_covid_mask | query_tb_mask
                query_mask_batch = np.stack([query_covid_mask, query_tb_mask], axis=1)
#                 query_mask_batch = np.stack([query_covid_mask, query_tb_mask, query_covid_tb_mask], axis=1)
                all_query_mask_batches.append(query_mask_batch)
        
            all_query_path_batches.append(query_path_batch)
            all_query_label_batches.append(query_label_batch)
            
            
            path_batch = np.array(path_batch)
            label_batch = np.array(label_batch).astype(int) ## categorical
            
            
            all_path_batches.append(path_batch)
            all_label_batches.append(label_batch)
                
        self.path_batches = np.stack(all_path_batches)  ## (batch_size, num_classes, num_samples_per_class)
        self.label_batches = np.stack(all_label_batches) ## current (batch_size, num_classes, num_samples_per_class, 2)
        
        self.query_path_batches = np.stack(all_query_path_batches) ## (batch_size, num_query)
        self.query_label_batches = np.stack(all_query_label_batches) ## (batch_size, num_classes, 2) 
        
        if self.eval:
            self.query_mask_batches = np.stack(all_query_mask_batches) ## (batch_size, 2)
            print(f'self.query_mask_batches.shape: {self.query_mask_batches.shape}') ## (batch_size, num_query, 2)

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()