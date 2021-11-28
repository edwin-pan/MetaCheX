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
                 shuffle_on_epoch_end=True, random_state=271, num_classes=27, multiclass=False,
                 parents_only=False):
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
        batch_y = self.y[idx * self.batch_size:end_idx]
        return batch_x, batch_y

    def load_image(self, image_path):
        image = Image.open(image_path)
        image_array = np.asarray(image.convert("RGB"))
        image_array = image_array / 255.
        image_array = resize(image_array, self.target_size)
        return image_array

    def get_y_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.
        """
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        
        end_idx = min(self.steps*self.batch_size, self.y.shape[0])
        return self.y[:self.steps*self.batch_size, :]

    def prepare_dataset(self):
        df = self.df.sample(frac=1., random_state=self.random_state)
        self.x_path = df['image_path']
        
        if self.parents_only: ## this is just to get parent embeddings later on
            self.y = np.eye(int(self.num_classes))[df['parent_id'].values.astype(int)]
        elif self.multiclass:
            self.y = np.eye(int(self.num_classes))[df['label_num_multi'].values.astype(int)]
        else:
            self.y = np.array(df['label_multitask'].to_list())
        print("self.y.shape: ", self.y.shape)

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()


class ProtoNetImageSequence(ImageSequence):
    
    def __init__(self, df, num_classes, num_samples_per_class, num_queries,
                 batch_size=8, target_size=(224, 224), verbose=0, steps=None, shuffle_on_epoch_end=True, 
                 random_state=271):
        
        self.df = df
        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class
        self.num_queries = num_queries
        self.batch_size = batch_size
        self.target_size = target_size
        self.verbose = verbose
        self.shuffle = shuffle_on_epoch_end
        self.random_state = random_state
        self.prepare_dataset()
        
        
    def __getitem__(self, idx):
        
        """
        batch_x: [num_classes x num_samples_per_class + num_queries, 224, 224, 3]
        batch_y: [num_classes x num_samples_per_class + num_queries, num_classes] (one-hot labels)
        """
        
        batch_x_path = self.path_batches[idx].flatten()
        batch_q = np.asarray([self.load_image(x_path) for x_path in self.path_query_batches[idx]]) 
        batch_x = np.asarray([self.load_image(x_path) for x_path in batch_x_path])
        batch_x = np.concatenate((batch_x, batch_q), axis=0)
        
        batch_y = np.eye(self.num_classes)[self.label_batches[idx].flatten()] 
        batch_q_label = np.eye(self.num_classes)[self.label_query_batches[idx]] 
        batch_y = np.concatenate((batch_y, batch_q_label), axis=0)
        
        return batch_x, batch_y
    
    def prepare_dataset(self, shuffle=False):
        """
        all_path_batches: [batch_size, num_classes, num_samples_per_class]
        all_label_batches: [batch_size, num_classes, num_samples_per_class]
        """
        
        all_path_batches, all_label_batches = [], []
        all_query_path_batches, all_query_label_batches = [], []
        
        for i in range(self.batch_size):
            ## sample num_classes from the classes in df
            all_classes = self.df['label_str'].drop_duplicates().values
            sampled_classes = np.random.choice(all_classes, self.num_classes)
            
            truncated_df = self.df[self.df['label_str'] in sampled_classes]
            
            ## for each of the num_classes, sample num_samples_per_class samples per class
            query_df = pd.DataFrame(columns=self.df.columns) 
            
            path_batch = []
            label_batch = []
            for j, classname in enumerate(sampled_classes):
                df_class = truncated_df[truncated_df['label_str'] == classname]
                df_class_sampled = df_class.sample(n=self.num_samples_per_class, random_state=self.random_state)
                
                df_class_query = pd.concat([df_class,df_class_sampled]).drop_duplicates(keep=False) 
                df_class_query['label'] = j
                query_df = query_df.append(df_class_query)
                
                path_batch.append(df_class_sampled['image_path'].values)
                label_batch.append([j] * self.num_samples_per_class)
            
            
            ## Sample queries
            sampled_query_df = query_df.sample(n=self.num_queries, random_state=self.random_state)
            query_path_batch = sampled_query_df['image_path'].to_numpy()
            query_label_batch = sampled_query_df['label'].to_numpy().astype(int) ## categorical
            all_query_path_batches.append(query_path_batch)
            all_query_label_batches.append(query_label_batch)
            
            path_batch = np.array(path_batch)
            label_batch = np.array(label_batch).astype(int) ## categorical
            
            batch = np.concatenate([path_batch, label_batch], 2)
            
#             ## shuffle matrix
#             if shuffle:
#                 for p in range(self.num_samples_per_class):
#                     np.random.shuffle(batch[:, p])
                    
            path_batch = batch[:, :, 0]
            label_batch = batch[:, :, 1]
            
            all_path_batches.append(path_batch)
            all_label_batches.append(label_batch)
                
        self.path_batches = np.stack(all_path_batches)
        self.label_batches = np.stack(all_label_batches)
        
        self.path_query_batches = np.stack(all_query_path_batches)
        self.path_label_batches = np.stack(all_query_label_batches)
        

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()