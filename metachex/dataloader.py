import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("error")

from glob import glob
from metachex.configs.config import *

class MetaChexDataset():
    def __init__(self):
        self.batch_size = 8
        
        # Reads in data.pkl file (mapping of paths to unformatted labels)
        self.data_path = os.path.join(PATH_TO_DATA_FOLDER, 'data.pkl')
        
        # Pre-processing
        print("[INFO] pre-processing")
        self.preprocess_data()
        self.df = pd.read_pickle(self.data_path)
        unique_labels_dict, df_combo_counts, df_label_nums, df_combo_nums = self.get_data_stats(self.df)

        # Trim low-sample count labels
        print("[INFO] truncating dataset")
        df_combos_exclude = df_combo_nums[df_combo_nums['count'] < SAMPLE_MIN].reset_index().rename(columns={'index': 'label_str'})
        self.df_condensed = self.df[~self.df['label_str'].isin(df_combos_exclude['label_str'])].reset_index(drop=True)
        self.unique_labels_dict, df_combo_counts, df_label_nums, df_combo_nums = self.get_data_stats(self.df_condensed)
#         print("Generated labels: ", len(list(self.unique_labels_dict.keys())))
        self.df_condensed = self.generate_labels(self.df_condensed, 'data_condensed.pkl')
        self.df_condensed['label_multitask'][1]

        print("[INFO] constructing tf train/val/test vars")
        [self.train_ds, self.val_ds, self.test_ds] = self.get_ds_splits(self.df_condensed)

        print("[INFO] shuffle & batch")
        self.train_ds = self.shuffle_and_batch(self.train_ds)
        self.val_ds = self.shuffle_and_batch(self.val_ds)

        print('[INFO] initialized')
        return


    def load_and_preprocess_image(self, path, label):
        """
        path: path to image
        """
        image = tf.io.read_file(path) 
        image_copy = tf.io.read_file(path)
        
        image_copy = tf.convert_to_tensor(image_copy)
        s = image_copy.shape
        if s.ndims is not None and s.ndims < 4:
            image = tf.io.decode_png(image, channels=1) 
            image = tf.image.grayscale_to_rgb(image)
            #grayscale
        else:
            image = tf.io.decode_png(image, channels=3)
       
        

#         try: 
#             print("Attempting to read rgb")
#             image = tf.io.decode_png(image, channels=3)
#         except Warning as e:
#             print("Image is grayscale. Decode to grayscale then convert to rgb")
#             image = tf.io.decode_png(image, channels=1) ## grayscale
#             image = tf.image.grayscale_to_rgb(image)

        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE], method='lanczos3')
        image = image / 255 ## pixels in [0, 255] -- normalize to [0, 1]
        return image, label


    def get_data_stats(self, df):
        unique_labels_dict = {} ## keys are str
        unique_combos_dict = {} ## keys are tuples of str
        for i in range(df.shape[0]):
            labels = df.at[i, 'label']
            for l in labels:
                if l not in unique_labels_dict:
                    unique_labels_dict[l] = 0
                unique_labels_dict[l] += 1

            label_str = df.at[i, 'label_str']
            if label_str not in unique_combos_dict:
                unique_combos_dict[label_str] = 0
            unique_combos_dict[label_str] += 1

        df_label_nums = pd.DataFrame.from_dict(unique_labels_dict, orient='index', columns=['count']).sort_values(by=['count'], ascending=False)
        df_combo_nums = pd.DataFrame.from_dict(unique_combos_dict, orient='index', columns=['count']).sort_values(by=['count'], ascending=False)

        ## Get number of labels with number of images in each range
        bins =  np.array([0, 10, 100, 1000, 10000, 100000])
        df_combo_counts = pd.DataFrame(columns=['count interval', 'number of labels'])
        df_combo_counts['count interval'] = ["< 5", "[5, 100)", "[100, 1k)", "[1k, 10k)", ">= 10k"]

        df_combo_counts['number of labels'] = [
                                    df_combo_nums[df_combo_nums['count'] < 5].size,
                                    df_combo_nums[(df_combo_nums['count'] >= 5) & (df_combo_nums['count'] < 1e2)].size,
                                    df_combo_nums[(df_combo_nums['count'] >= 1e2) & (df_combo_nums['count'] < 1e3)].size,
                                    df_combo_nums[(df_combo_nums['count'] >= 1e3) & (df_combo_nums['count'] < 1e4)].size,
                                    df_combo_nums[df_combo_nums['count'] >= 1e4].size
                                    ]
            
        return unique_labels_dict, df_combo_counts, df_label_nums, df_combo_nums


    ## Train-val-test splits
    def get_ds_splits(self, df, split=(0.7, 0.1, 0.2)):
        """
        df: df of nih data; columns: 'image_path', 'label', 'dataset', 'label_str', 'label_multitask'
        """
        
        ## Deal with NIH datasplit first
        nih_datasets = []
        nih_ds_sizes = []
        for ds_type in ['train', 'val', 'test']:
            df_nih = df[df['dataset'] == ds_type]
            ds = tf.data.Dataset.from_tensor_slices((df_nih['image_path'], 
                                                    df_nih['label_multitask'].to_list()))
            nih_ds_sizes.append(len(ds))
            nih_datasets.append(ds)
        
        
        print("NIH ds sizes", nih_ds_sizes)
        # return nih_datasets

        ## Non-nih data
        df_other = df[df['dataset'].isna()]
        other_count = len(df_other)
        total_count = len(df)
        ds = tf.data.Dataset.from_tensor_slices((df_other['image_path'], df_other['label_multitask'].to_list()))
        ds = ds.shuffle(other_count, reshuffle_each_iteration=False)
        train_count = max(int(total_count * split[0]) - nih_ds_sizes[0], 0)
        val_count = max(int(total_count * split[1]) - nih_ds_sizes[1], 0)
        
        train_ds = ds.take(train_count)
        val_ds = ds.skip(train_count).take(val_count)
        test_ds = ds.skip(train_count + val_count) 
        
        other_datasets = [train_ds, val_ds, test_ds]
        other_ds_sizes = [len(d) for d in other_datasets]
        print("Other ds sizes", other_ds_sizes)
        
        full_datasets = []
        total_samples = 0
        for i in range(3):
            ds = other_datasets[i].concatenate(nih_datasets[i])
            print(len(ds))
            full_datasets.append(ds)
            total_samples += len(ds)
            
        print("Total samples: ", total_samples)
        
        full_ds_split = [len(d) / total_count for d in full_datasets]
        print("True split: ", full_ds_split)
        self.steps_per_epoch = (len(full_datasets[0]) / self.batch_size) * 0.1
        
        return full_datasets


    def preprocess_data(self):
        """
        If data.pkl exists, read from it.

        Otherwise, extract filenames and labels for:
        - ChestX-ray14 (NIH) dataset
        - COVID-19 Radiography Dataset
        - covid-chestxray-dataset

        and put in data.pkl
        """

        if os.path.isfile(self.data_path): # path does exist
            print(f"Data already processed. Loading from save {self.data_path}")
        else:
            print(f"Data needs to be processed. Proceeding...")
            df = pd.DataFrame(columns=['image_path', 'label', 'dataset'])

            ## NIH
            full_path = os.path.join(PATH_TO_DATA_FOLDER, NIH_METADATA_PATH)
            df_nih = pd.read_csv(full_path)[['Image Index', 'Finding Labels']]
            df_nih.rename(columns={'Image Index': 'image_path', 'Finding Labels': 'label'}, inplace=True)
            
            # Keep only 20k 'No Finding' images
            df_nih_no_finding = df_nih[df_nih['label'] == 'No Finding'] # get all no findings
            df_nih_no_finding = df_nih_no_finding.sample(n=10000, random_state=271) # choose only 20k
            
            df_nih = df_nih[df_nih['label'] != 'No Finding'] ## remove all no findings
            df_nih = df_nih.append(df_nih_no_finding)
            
            df_nih['label'] = df_nih['label'].str.strip().str.split('|')
            
            # --- Denotes which dataset (train/val/test) the images belong to
            df_nih_splits = pd.DataFrame(columns=['image_path', 'dataset'])
            for dataset_type in ['train', 'val', 'test']:
                sub_df_nih = pd.read_csv(os.path.join('CheXNet_data_split', f'{dataset_type}.csv'), 
                                         usecols=['Image Index']).rename(columns={'Image Index': 'image_path'})
                sub_df_nih['dataset'] = dataset_type
                df_nih_splits = df_nih_splits.append(sub_df_nih)
            
            df_nih = df_nih.merge(df_nih_splits, how='left')
            ## ----- 
            
            df_nih['image_path'] = PATH_TO_DATA_FOLDER + '/' + NIH_IMAGES + '/' + df_nih['image_path']
            df = df.append(df_nih)

            ## COVID_CHESTXRAY
            full_path = os.path.join(PATH_TO_DATA_FOLDER, COVID_CHESTXRAY_METADATA_PATH)
            df_cc = pd.read_csv(full_path)[['filename', 'finding']]
            df_cc.rename(columns={'filename': 'image_path', 'finding': 'label'}, inplace=True)
            df_cc = df_cc.drop(df_cc[(df_cc['label'] == 'todo') | (df_cc['label'] == 'Unknown')].index).reset_index(drop=True)
            df_cc['label'] = df_cc['label'].str.strip()
            ## Remove all the 'No Finding' images
            df_cc = df_cc[df_cc['label'] != 'No Finding']
            df_cc['label'] = df_cc['label'].str.split('/')
            df_cc = df_cc.reset_index(drop=True)
            ## Remove the label after 'Pneumonia' that specifies type of pneumonia if given
            for i in range(df_cc.shape[0]):
                label = df_cc.at[i, 'label']
                if 'Pneumonia' in label and len(label) > 1:
                    p_idx = label.index('Pneumonia')
                    label.pop(p_idx + 1)
                    #sort the labels to be in alphabetical order
                    df_cc.at[i, 'label'] = sorted(label)
            
            df_cc['image_path'] = PATH_TO_DATA_FOLDER + '/' + COVID_CHESTXRAY_IMAGES + '/' + df_cc['image_path']
            
            ## Remove all images that have .gz extension
            df_cc = df_cc[df_cc['image_path'].str[-2:] != 'gz']
            
            df = df.append(df_cc)

            ## COVID-19 Radiography
            full_path = os.path.join(PATH_TO_DATA_FOLDER, COVID_19_RADIOGRAPHY_IMAGES)
            df_cr = pd.DataFrame(columns=['image_path', 'label'])
            image_lst = sorted(list(glob(f"{full_path}/*"))) ## gets list of all image filepaths
            label_arr = np.array([f[f.rindex('/') + 1:f.rindex('-')] for f in image_lst])
            label_arr = np.where(label_arr == 'COVID', 'COVID-19', label_arr) ## replace COVID with COVID-19 for consistency
            label_arr = np.where(label_arr == 'Viral Pneumonia', 'Pneumonia', label_arr)
            df_cr['image_path'] = image_lst
            df_cr['label'] = label_arr
            
            ## Remove all the 'Normal' images
            df_cr = df_cr[df_cr['label'] != 'Normal']
            
            ## makes each label a list (random sep so that no split on space)
            df_cr['label'] = df_cr['label'].str.strip().str.split(pat='.') 
            df = df.append(df_cr)

            df = df.reset_index(drop=True)
            df['label'] = df['label'].sort_values().apply(lambda x: sorted(x)) ## final sort just in case
            df['label_str'] = df['label'].str.join('|')
            df.to_pickle(self.data_path)
            print(f"Saved {self.data_path}")
        return


    def generate_labels(self, df, filename, combo=True):
        """
        Generates the multiclass (categorical) and binary multitask (list of 0's and 1's) labels
        df: DataFrame of paths to unformatted labels
        filename: where to save new dataframe to (pkl)
        combo: whether or not to generate the multiclass (categorical) labels
        """
        path = os.path.join(PATH_TO_DATA_FOLDER, filename)
        if not os.path.isfile(path): ## path does not exist
            if combo:
                ## Get combo label (for multiclass classification)
                df['label_num_multi'] = df.groupby(['label_str']).ngroup()

            ## Get binary multi-task labels
            unique_labels = list(self.unique_labels_dict.keys())
            unique_labels.remove('No Finding')
            unique_labels.sort() ## alphabetical order

            df['label_multitask'] = 0
            df['label_multitask'] = df['label_multitask'].astype('object')
            for i, row in df.iterrows():
                indices = []
                for l in row['label']:
                    if l == 'No Finding':
                        break

                    idx = unique_labels.index(l)
                    indices.append(idx)

                if indices == []:
                    df.at[i, 'label_multitask'] = np.zeros(len(unique_labels)).astype(np.uint8)
                else:
                    df.at[i, 'label_multitask'] = np.eye(len(unique_labels))[indices].sum(axis=0).astype(np.uint8)

            ## Save to disk
            df.to_pickle(path)
        else:
            df = pd.read_pickle(path)
        
        return df
    
    
    def get_class_weights(self, one_cap=False):
        """ Compute class weighting values for dataset (both individual and combo labels computed)."""
        _, _, indiv, combo = self.get_data_stats(self.df_condensed)
        if one_cap: # Restrains between 0 and 1
            indiv_weights = (indiv['count'].sum() - indiv['count'])/indiv['count'].sum()
            # indiv_prob_class = (indiv['count'])/indiv['count'].sum()
            combo_weights = (combo['count'].sum() - combo['count'])/combo['count'].sum()
        else: # Unconstrained
            indiv_weights = (1 / indiv['count']) * (indiv['count'].sum() / indiv.shape[0]) # weight we want to apply if class is TRUE
            indiv_weights_false = (1 / (indiv['count'].sum()-indiv['count'])) * (indiv['count'].sum() / indiv.shape[0]) # weight we want to apply if class is TRUE
            combo_weights = (1 / combo['count']) * (combo['count'].sum() / combo.shape[0])
        
        indiv_weights = indiv_weights.sort_index()
        indiv_weights = indiv_weights.drop(['No Finding'])
        indiv_weights_false = indiv_weights_false.sort_index()
        indiv_weights_false = indiv_weights_false.drop(['No Finding'])
        combo_weights = combo_weights.sort_index()
        
        indiv_class_weights = dict(list(enumerate((indiv_weights.values, indiv_weights_false.values))))
        combo_class_weights = dict(list(enumerate(combo_weights.values)))
        
        return np.array([indiv_weights.values, indiv_weights_false.values]), indiv_class_weights, combo_class_weights
    

    def get_class_probs(self):
        """ Compute class probabilities for dataset (both individual and combo labels computed)"""
        _, _, indiv, combo = self.get_data_stats(self.df_condensed)
        indiv_class_probs = indiv['count']/indiv['count'].sum()
        combo_class_probs = combo['count']/combo['count'].sum()
        
        return indiv_class_probs, combo_class_probs
    

    def shuffle_and_batch(self, ds):
        """ Train/val/test split: Remember that the NIH data has to be split according to how it was pre-trained. """
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=100)
        ds = ds.map(self.load_and_preprocess_image) ## maps the preprocessing step
        ds = ds.batch(self.batch_size)
#         ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds


if __name__ == '__main__':
    dataset = MetaChexDataset()
    train_ds = dataset.train_ds
    val_ds = dataset.val_ds
    test_ds = dataset.test_ds

    # Grab one sample
    next(iter(train_ds))
