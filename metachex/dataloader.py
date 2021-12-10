import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import warnings
import pickle
warnings.filterwarnings("error")

from glob import glob

from metachex.configs.config import *

from PIL import Image
import sklearn
from skimage.transform import resize

from metachex.image_sequence import ImageSequence, ProtoNetImageSequence
import pickle5 as pickle

np.random.seed(271)

class MetaChexDataset():
    def __init__(self, shuffle_train=True, multiclass=False, baseline=False, protonet=False, batch_size=8, 
                 n=5, k=3, n_query=5, n_test=5, k_test=3, n_test_query=5,
                 num_meta_train_episodes=100, num_meta_val_episodes=1, num_meta_test_episodes=1000,
                 max_num_vis_samples=100):
        self.batch_size = batch_size
        self.multiclass = multiclass
        
        # Reads in data.pkl file (mapping of paths to unformatted labels)
        self.data_path = os.path.join(PATH_TO_DATA_FOLDER, 'data.pkl')
        
        # Datasplit path
        self.datasplit_path = os.path.join(PATH_TO_DATA_FOLDER, 'datasplit.pkl')
        
        ## Child to parent map and num_parents_list path
        self.child_to_parent_map_path = os.path.join(PATH_TO_DATA_FOLDER, 'childParent.pkl')
        self.num_parents_list_path = os.path.join(PATH_TO_DATA_FOLDER, 'num_parents_list.pkl')
        self.parent_multiclass_labels_path = os.path.join(PATH_TO_DATA_FOLDER, 'parent_multiclass_labels.npy')
        self.tsne1_ds_path = os.path.join(PATH_TO_DATA_FOLDER, 'tsne1_ds.pkl')
        self.tsne2_ds_path = os.path.join(PATH_TO_DATA_FOLDER, 'tsne2_ds.pkl')
        
        
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
        self.df_condensed = self.generate_labels(self.df_condensed, 'data_condensed.pkl')
        
        if multiclass or protonet:
            self.df_parents, self.df_covid_tb, self.df_children = self.subsample_data()

        print("[INFO] constructing tf train/val/test vars")
         ## already shuffled and batched
        print("[INFO] shuffle & batch")
        if protonet:
            self.num_meta_train_episodes = num_meta_train_episodes
            self.num_meta_val_episodes = num_meta_val_episodes
            self.num_meta_test_episodes = num_meta_test_episodes
            # Note: test_ds includes labels in self.df_parents as well as covid and tb
            [self.train_ds, self.val_ds, self.test_ds] = self.get_protonet_generator_splits2(self.df_parents, self.df_covid_tb, 
                                                                                            n, k, n_query, n_test, k_test,
                                                                                            n_test_query,
                                                                                            shuffle_train=shuffle_train)
            self.tsne1_ds, self.tsne2_ds = self.get_tsne_generators(max_num_vis_samples)
            self.n, self.k, self.n_query = n, k, n_query
            self.n_test, self.k_test, self.n_test_query = n_test, k_test, n_test_query
        elif multiclass:
            if not baseline:
                [self.train_ds, self.test_ds] = self.get_multiclass_generator_splits(self.df_condensed, 
                                                                                     shuffle_train=shuffle_train)
            else:
                self.df_combined = self.df_parents.append(self.df_covid_tb).reset_index(drop=True)
                [self.train_ds, self.val_ds, self.test_ds] = self.get_multiclass_generator_splits(self.df_combined,
                                                                                                  shuffle_train=shuffle_train,
                                                                                                  baseline=baseline)
                self.tsne1_ds, self.tsne2_ds = self.get_tsne_generators(max_num_vis_samples)
        else:
            print('Must choose protonet or multiclass')
            exit(1)

        print('[INFO] initialized')
        return

    
    def get_tsne_generators(self, max_num_samples_per_class=100):
        """
        Returns image sequences corresponding to the two main relationships we want to visualize
        
        1. Parents and their children: Atelectasis, Effusion, Infiltration, all combos
        2. Similar vs. distant classes: COVID-19, Pneumonia, TB (“similar”); Mass (“distant”); No Finding (bonus)
        """
        
        tsne1_ds, tsne2_ds = None, None
        
        ## Load from pickle if exist
        if os.path.exists(self.tsne1_ds_path):
            with open(self.tsne1_ds_path, 'rb') as file:
                tsne1_ds = pickle.load(file)
        
        if os.path.exists(self.tsne2_ds_path):
            with open(self.tsne2_ds_path, 'rb') as file:
                tsne2_ds = pickle.load(file)
        
        if tsne1_ds is not None and tsne2_ds is not None:
            return tsne1_ds, tsne2_ds
        
        ## 1. Parents + their children
        test_df = self.test_ds.df.append(self.df_covid_tb)
        test_df = test_df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
        
        print(test_df['label_str'].drop_duplicates().values)
        
        ## Get parents
        df_parents = pd.DataFrame(columns=test_df.columns)
        for parent in TSNE_PARENT_CLASSES:
            df_class = test_df[test_df['label_str'] == parent] # Get class rows
            df_class = df_class.sample(n=min(max_num_samples_per_class, len(df_class))) # Sample
            df_parents = df_parents.append(df_class) # Append
        
        ## Get children
        df_children = pd.DataFrame(columns=self.df_children.columns)
        for child in TSNE_CHILD_CLASSES:
            df_class = self.df_children[self.df_children['label_str'] == child] # Get class rows
            df_class = df_class.sample(n=min(max_num_samples_per_class, len(df_class))) # Sample
            df_children = df_children.append(df_class) # Append
        
        ## Combine parents + children
        df_parents_and_children = df_parents.append(df_children).reset_index(drop=True)
        
        ## 2. Similar vs. distant
        df_distance = pd.DataFrame(columns=test_df.columns)
        for label in TSNE_DISTANCE_CLASSES:
            df_class = test_df[test_df['label_str'] == label] # Get class rows
            df_class = df_class.sample(n=min(max_num_samples_per_class, len(df_class))) # Sample
            df_distance = df_distance.append(df_class) # Append
        
        df_distance = df_distance.reset_index(drop=True)
        print(df_distance['label_str'])
        
        ## Return image sequences
        tsne1_ds = ImageSequence(df=df_parents_and_children, shuffle_on_epoch_end=False, 
                               num_classes=self.num_classes_multiclass, multiclass=True, batch_size=self.batch_size,
                                tsne=True)
        tsne2_ds = ImageSequence(df=df_distance, shuffle_on_epoch_end=False, 
                               num_classes=self.num_classes_multiclass, multiclass=True, batch_size=self.batch_size,
                                tsne=True)
        
        ## Dump to pickle
        with open(self.tsne1_ds_path, 'wb') as file:
            pickle.dump(tsne1_ds, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(self.tsne2_ds_path, 'wb') as file:
            pickle.dump(tsne2_ds, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        return tsne1_ds, tsne2_ds
    
    def get_protonet_generator_splits2(self, df, df_held_out, n, k, n_query, n_test, k_test, n_test_query, 
                                      split=(0.7, 0.1, 0.2), shuffle_train=True):
        """
        Get datasplits (the same as the multiclass baseline but with added covid_tb to meta-test)
        """
        
        data_splits = self.get_data_splits3(df, split=split) ## (train, val, test)
        
        ds_types = ['train', 'val', 'test']
        
        datasets = []
        for i, ds_type in enumerate(ds_types):
            shuffle_on_epoch_end = False
            num_classes, num_samples_per_class, num_queries = n, k, n_query
            if ds_type == 'train':
                shuffle_on_epoch_end = True
                steps = self.num_meta_train_episodes 
                eval = False
            elif ds_type == 'test':
                num_classes, num_samples_per_class, num_queries = n_test, k_test, n_test_query
                steps = self.num_meta_test_episodes 
                
                data_splits[i] = data_splits[i].append(df_held_out)
                data_splits[i] = data_splits[i].sample(frac=1).reset_index(drop=True)
                eval = True
                
            else: # val
                steps = self.num_meta_val_episodes
                eval = False
            
#             print(ds_type)
            ds = ProtoNetImageSequence(data_splits[i], steps=steps, num_classes=num_classes, 
                                       num_samples_per_class=num_samples_per_class, 
                                       num_queries=num_queries, batch_size=self.batch_size, 
                                       shuffle_on_epoch_end=shuffle_on_epoch_end)
            
            datasets.append(ds)
        return datasets
        
    
    def get_data_stats2(self, df):
        """
        returns label to count df
        """
        df_counts = df.groupby(['label_str']).size().to_frame('count')
        total = df_counts['count'].values.sum()
        print(f'Total number of images for 18-way classification: {total}')
        return df_counts
    
    
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
            
        total = df_combo_nums['count'].values.sum()
        print(f'Total number of images: {total}')
        df_combo_counts.to_csv('data/df_combo_counts.csv', index=False)
        df_label_nums.to_csv('data/df_label_nums.csv', index=True)
        df_combo_nums.to_csv('data/df_combo_nums.csv', index=True)

        return unique_labels_dict, df_combo_counts, df_label_nums, df_combo_nums
    
    
    def get_data_splits3(self, df, split=(0.7, 0.1, 0.2)):
        """
        Splits according to multiclass label to the split percentages as best as possible
        Note: We DO NOT split according to pre-defined NIH splits, since we will train from scratch
        """
        # Load datasplit if it exists
        if os.path.isfile(self.datasplit_path): 
            with open(self.datasplit_path, 'rb') as file:
                data_splits = pickle.load(file)
        
            return data_splits
        
        # Datasplit does not exist
        
        ## Split the rest of the data relatively evenly according to the ratio per class
        ## That is, for each label, the first 70% goes to train, the next 20% to val, the final 10% to test
        df = sklearn.utils.shuffle(df, random_state=271) # shuffle
        
        data_splits = [pd.DataFrame(columns=df.columns)] * 3
        
        image_paths, multiclass_labels = df['image_path'].values, df['label_num_multi'].values
        
        data_dict = {'label': [], 'count': []}
        
        for i in range(df['label_num_multi'].max() + 1):
            rows_with_label = np.where(multiclass_labels == i)
            images_with_label = image_paths[rows_with_label]
            
            df_subsplit = df.loc[df['image_path'].isin(images_with_label)].reset_index(drop=True)

            if len(df_subsplit) > 0:
                label = df_subsplit['label_str'].drop_duplicates().values[0]
                data_dict['label'].append(label)
                data_dict['count'].append(len(df_subsplit))
                
            
            val_idx = int(len(df_subsplit) * split[0])
            test_idx = val_idx + int(len(df_subsplit) * split[1])
            data_splits[0] = data_splits[0].append(df_subsplit[:val_idx])
            data_splits[1] = data_splits[1].append(df_subsplit[val_idx:test_idx])
            data_splits[2] = data_splits[2].append(df_subsplit[test_idx:])
        
        data_counts = pd.DataFrame.from_dict(data_dict)
        data_counts.to_csv(os.path.join(PATH_TO_DATA_FOLDER, 'data_counts.csv'), index=False)
        print(f'data_counts: \n {data_counts}')
            
        print(f'split counts: {[len(data_splits[i]) for i in range(len(data_splits))]}')
        
        ## Dump to pickle
        with open(self.datasplit_path, 'wb') as file:
            pickle.dump(data_splits, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        return data_splits
    
    
    def get_multiclass_generator_splits(self, df, split=(0.7, 0.1, 0.2), shuffle_train=True, baseline=False):
        """
        Splitting with tensorflow sequence instead of dataset
        
        Note: split is a 2-length tuple iff baseline == False; otherwise, 3-length
        """
        
#         data_splits = self.get_data_splits2(df, split=(split[0] // 2, split[0] - split[0] // 2, split[1])) ## (train, val, test)
        data_splits = self.get_data_splits3(df, split=split) ## (train, val, test)
        
        if not baseline: ## combine train and val
            data_splits = [data_splits[0].append(data_splits[1]).reset_index(drop=True), data_splits[2]]
        
        full_datasets = []
        
        if not baseline:
            ds_types = ['train', 'test']
        else:
            ds_types = ['train', 'val', 'test']
            
        # -------------------------------------------------
        ## Getting information about the splits
        df_combined = data_splits[0]
        df_multiclass_labels = np.array(df_combined['label_num_multi'].to_list())
        df_multiclass_labels = np.eye(self.num_classes_multiclass)[np.array(df_multiclass_labels)]
        df_multiclass_labels_sum = np.sum(df_multiclass_labels, axis=0)
        untrained_classes = np.where(df_multiclass_labels_sum == 0)
        print(f'{untrained_classes[0].shape[0]} classes not trained on: {untrained_classes}')

        df_combined = data_splits[1]
        df_multiclass_labels = np.array(df_combined['label_num_multi'].to_list())
        df_multiclass_labels = np.eye(self.num_classes_multiclass)[np.array(df_multiclass_labels)]
        df_multiclass_labels_sum = np.sum(df_multiclass_labels, axis=0)
        unvalidated_classes = np.where(df_multiclass_labels_sum == 0)
        
        if baseline:
            print(f'{unvalidated_classes[0].shape[0]} classes not validated on: {unvalidated_classes}')

            df_combined = data_splits[2]
            df_multiclass_labels = np.array(df_combined['label_num_multi'].to_list())
            df_multiclass_labels = np.eye(self.num_classes_multiclass)[np.array(df_multiclass_labels)]
            df_multiclass_labels_sum = np.sum(df_multiclass_labels, axis=0)
            untested_classes = np.where(df_multiclass_labels_sum == 0)
            print(f'{untested_classes[0].shape[0]} classes not tested on: {untested_classes}')
        else:
            print(f'{unvalidated_classes[0].shape[0]} classes not tested on: {unvalidated_classes}')
        # -------------------------------------------------
        
        for i, ds_type in enumerate(ds_types):
            
            if ds_type == 'train':
                shuffle_on_epoch_end = shuffle_train
                factor = 0.1
            elif ds_type == 'val':
                shuffle_on_epoch_end = False
                factor = 0.2
            else: ## test
                shuffle_on_epoch_end = False
                factor = 1
            
            steps = int(len(df_combined) / self.batch_size * factor)
            
            if ds_type == 'train':
                self.train_steps_per_epoch = steps
            elif ds_type == 'val':
                self.val_steps_per_epoch = steps
            else: ## test
                steps += 1 if len(df_combined) > self.batch_size * factor else 0 ## add for extra incomplete batch
                
            ds = ImageSequence(df=data_splits[i], steps=steps, shuffle_on_epoch_end=shuffle_on_epoch_end, 
                               num_classes=self.num_classes_multiclass, multiclass=True, batch_size=self.batch_size)
            full_datasets.append(ds)
        
        return full_datasets
   
    
    def subsample_data(self):
        """
        self.df_condensed: the already truncated df with multitask labels
        
        Return: 
        df_parents: df of all classes that appear individually as well as no finding, except covid and tb
        df_covid_tb: df of just covid and tb samples
        df_children: df of children of parents who appear individually (ie parents df_parents and/or df_covid_tb)
        """
        
        ## Get only rows with singular classes (ie | not in df['label_str']
        # Note: \| to escape the |
        df_parents = self.df_condensed[~self.df_condensed['label_str'].str.contains('\|')].reset_index(drop=True)
        df_parent_labels = df_parents[['label_str', 'label_multitask']].drop_duplicates(subset=['label_str'])
        self.num_classes_multiclass = len(df_parent_labels) # note: includes no finding
        df_parents['label_num_multi'] = df_parents.groupby(['label_str']).ngroup() # reset multiclass labels
        
        label_multitask_arr = np.array(df_parent_labels['label_multitask'].to_list()) ## [# indiv parents, 27]
        
        ## get parent rows and their multitask indices
        _, parent_multitask_indices = np.where(label_multitask_arr == 1)
        
        ## Get all possible children (complement of df_parents)
        df_children = self.df_condensed[self.df_condensed['label_str'].str.contains('\|')].reset_index(drop=True)
        
        ## Truncate df_children to only the ones whose parents are in df_parents
        label_multitask_arr = np.array(df_children['label_multitask'].to_list()) ## [len(df_children), num_indiv_classes]
        print(label_multitask_arr.shape)
        child_row_indices, child_multitask_indices = np.where(label_multitask_arr == 1)

        child_row_indices_we_want = child_row_indices[np.in1d(child_multitask_indices, parent_multitask_indices)]
        
        df_children = df_children.iloc[np.unique(child_row_indices_we_want)]
        
        ## Labels
        self.unique_labels = df_parents.sort_values(by=['label_num_multi'])['label_str'].drop_duplicates().values
        
        ## Separate df_parents into covid_tb and everything else
        df_covid_tb = df_parents[df_parents['label_str'].isin(['COVID-19', 'Tuberculosis'])]
        df_parents = df_parents[~df_parents['label_str'].isin(['COVID-19', 'Tuberculosis'])]
        
        return df_parents, df_covid_tb, df_children
        

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
            df_nih_no_finding = df_nih_no_finding.sample(n=20000, random_state=271) # choose only 20k
            
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
            self.unique_labels = unique_labels
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
            
            self.unique_labels = sorted(list(self.unique_labels_dict.keys()))
            self.unique_labels.remove('No Finding')
            print(len(self.unique_labels))
        
        self.num_classes_multitask = len(self.unique_labels)
        self.num_classes_multiclass = max(df['label_num_multi'].values) + 1
        
        if self.multiclass:
            label_df = df[['label_num_multi', 'label_str']].drop_duplicates().sort_values('label_num_multi')
            self.unique_labels = label_df.sort_values(by=['label_num_multi'])['label_str'].values
            
        return df
    
    
    def get_class_weights2(self, one_cap=False):
        """
        Gets class weights for baseline multiclass (18-way)
        """
        counts = self.get_data_stats2(self.df_combined)
        if one_cap: # Restrains between 0 and 1
            weights = (counts['count'].sum() - counts['count'])/counts['count'].sum()
            weights_false = counts['count']/counts['count'].sum()
        else: # Unconstrained
            weights = (1 / counts['count']) * (counts['count'].sum() / counts.shape[0])
            weights_false = (1 / (counts['count'].sum()-counts['count'])) * (counts['count'].sum() / counts.shape[0]) 
        
        weights = weights.sort_index()
        weights_false = weights_false.sort_index()
        
        return np.array([weights.values, weights_false.values])
