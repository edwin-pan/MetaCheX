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
# from configs.config import *

from PIL import Image
import sklearn
from skimage.transform import resize

from metachex.image_sequence import ImageSequence, ProtoNetImageSequence

np.random.seed(271)

class MetaChexDataset():
    def __init__(self, shuffle_train=True, multiclass=False, baseline=False, protonet=False, batch_size=8, 
                 n=5, k=3, n_query=5, n_test=5, k_test=3, n_test_query=5,
                 num_meta_train_episodes=100, num_meta_test_episodes=1000):
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
        
        if multiclass or protonet:
            print("[INFO] get child-to-parents mapping")
            ## list of multiclass labels corresponding to multitask index
            self.parent_multiclass_labels = np.ones((self.num_classes_multitask + 1,)) * -1  ## +1 for 'No Finding' label
            self.get_child_to_parent_mapping()
            self.get_num_parents_to_count_df()

        print("[INFO] constructing tf train/val/test vars")
         ## already shuffled and batched
        print("[INFO] shuffle & batch")
        if protonet:
            self.num_meta_train_episodes = num_meta_train_episodes
            self.num_meta_test_episodes = num_meta_test_episodes
            [self.train_ds, self.val_ds, self.test_ds] = self.get_protonet_generator_splits(self.df_condensed, 
                                                                                            n, k, n_query, n_test, k_test,
                                                                                            n_test_query,
                                                                                            shuffle_train=shuffle_train)
            self.n, self.k, self.n_query = n, k, n_query
            self.n_test, self.k_test, self.n_test_query = n_test, k_test, n_test_query
        elif multiclass:
            if not baseline:
                [self.train_ds, self.test_ds] = self.get_multiclass_generator_splits(self.df_condensed, 
                                                                                     shuffle_train=shuffle_train)
            else:
                [self.train_ds, self.val_ds, self.test_ds] = self.get_multiclass_generator_splits(self.df_condensed,
                                                                                                  shuffle_train=shuffle_train,
                                                                                                  baseline=baseline)
        else:
            [self.train_ds, self.val_ds, self.test_ds] = self.get_multitask_generator_splits(self.df_condensed, 
                                                                                             shuffle_train=shuffle_train)

        print('[INFO] initialized')
        return

    
    def get_protonet_generator_splits(self, df, n, k, n_query, n_test, k_test, n_test_query, 
                                      split=(0.8, 0.1, 0.1), shuffle_train=True):
        """
        Get datasets for train, val and test (n-way, k-shot)
        """
        
        ## Get labels for train, val, test
        train_num = int(self.num_classes_multiclass * split[0])
        val_num = int(self.num_classes_multiclass * split[1])
        
        unique_label_strs = df['label_str'].drop_duplicates().values
        np.random_seed(271)
        np.random.shuffle(unique_label_strs)
        
        train_label_strs = unique_label_strs[:train_num]
        val_label_strs = unique_label_strs[train_num : train_num + val_num]
        test_label_strs = unique_label_strs[train_num + val_num :]
        
        label_strs = [train_label_strs, val_label_strs, test_label_strs]
        
        ds_types = ['train', 'val', 'test']
        
        dfs = [df[df['label_str'].isin(lst)].reset_index(drop=True) for lst in label_strs]
        
        datasets = []
        for i, ds_type in enumerate(ds_types):
            shuffle_on_epoch_end = False
            num_classes, num_samples_per_class, num_queries = n, k, n_query
            if ds_type == 'train':
                shuffle_on_epoch_end = True
                steps = self.num_meta_train_episodes 
            elif ds_type == 'test':
                num_classes, num_samples_per_class, num_queries = n_test, k_test, n_test_query
                steps = self.num_meta_test_episodes 
            else: # val
                steps = 1 
            
            ds = ProtoNetImageSequence(dfs[i], steps=steps, num_classes=num_classes, 
                                       num_samples_per_class=num_samples_per_class, 
                                       num_queries=num_queries, batch_size=self.batch_size, 
                                       shuffle_on_epoch_end=shuffle_on_epoch_end)
            
            datasets.append(ds)
        return datasets
    
        
    def get_num_parents_to_count_df(self):
        
        df = pd.DataFrame(columns=['num_parents', 'num_examples'])
        
        df['num_parents'] = np.unique(self.num_parents_list)
        
        counts = []
        for num in df['num_parents'].values:
            count = np.where(self.num_parents_list == num)[0].shape[0]
            counts.append(count)
            
        df['num_examples'] = counts
        
        print(df)
    
    
    def get_child_to_parent_mapping(self):
        """
        Get dict of child index to list of parent indices for all child classes
        
        returns: {multiclass_child_ind (0 to 328) : list of multitask parent indices (0 to 26)}
        """
        
        ## load maps and lists
        if os.path.isfile(self.child_to_parent_map_path) and os.path.isfile(self.parent_multiclass_labels_path): 
            with open(self.child_to_parent_map_path, 'rb') as file:
                self.child_to_parent_map = pickle.load(file)
            
            with open(self.num_parents_list_path, 'rb') as file:
                self.num_parents_list = pickle.load(file)
            
            self.parent_multiclass_labels = np.load(self.parent_multiclass_labels_path)
            return
        
        ## Roundabout way to get a copy of df_condensed (since col 'multitask_indices' has lists)
        df_labels = pd.DataFrame(self.df_condensed['label_num_multi'].values, columns=['label_num_multi'])
        df_labels['label_str'] = self.df_condensed['label_str'].values
        
        ## Get all the multitask indices for each row
        label_multitask_arr = np.array(self.df_condensed['label_multitask'].to_list()) ## [len(df_condensed), num_indiv_classes]
        row_indices, multitask_indices = np.where(label_multitask_arr == 1)

        df_labels['multitask_indices'] = 0
        df_labels['multitask_indices'] = df_labels['multitask_indices'].astype(object)
        
        num_parents = [] ## list of number of parents for each row
        
        for i, row in df_labels.iterrows():
            ## get list of multitask indices associated with the multiclass label
            indices_for_row = np.where(row_indices == i)
            multitask_indices_for_row = multitask_indices[indices_for_row]
            df_labels.at[i, 'multitask_indices'] = multitask_indices_for_row
            num_parents.append(multitask_indices_for_row.shape[0])

            ## Gets multitask to multiclass mapping for parents that exist individually
            if multitask_indices_for_row.shape[0] == 1: # indiv class
                parent_label_num_multi = row['label_num_multi']
                if self.parent_multiclass_labels[multitask_indices_for_row[0]] == -1:
                    self.parent_multiclass_labels[multitask_indices_for_row[0]] = parent_label_num_multi
        
        ## Note: parents who do not exist individually will be marked by a -1 in the self.parent_multiclass_labels array
        
        ## Add 'No Finding' multiclass label to self.parent_multiclass_labels
        no_finding_label = df_labels['label_num_multi'][df_labels['label_str'] == 'No Finding'].drop_duplicates().values[0]               
        self.parent_multiclass_labels[-1] = no_finding_label
        
        ## Populate the rest of the self.parent_multiclass_labels (parents that only exist as combos)
        ## I realized that this is not necessary, but just in case we somehow need it
#         indiv_parent_indices = np.where(self.parent_multiclass_labels == -1)[0]
#         self.parent_multiclass_labels[indiv_parent_indices] = np.arange(indiv_parent_indices.shape[0]) \
#                                                                 + self.num_classes_multiclass
        
#         print(self.parent_multiclass_labels)

        self.num_parents_list = num_parents
    
        ## Dump to pickle
        with open(self.num_parents_list_path, 'wb') as file:
            pickle.dump(self.num_parents_list, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        np.save(self.parent_multiclass_labels_path, self.parent_multiclass_labels)
        
        ## Get the parent multiclass labels if the indiv class exists
        child_to_parent_map = {}

        for i, row in df_labels.iterrows():
            parents = np.array([])
            if row['multitask_indices'].shape[0] > 1: ## combo
                parents = row['multitask_indices']
                
#                 for ind in row['multitask_indices']:
#                     if ind in indiv_parent_multitask_to_multiclass:
#                         parent_label_num_multi = indiv_parent_multitask_to_multiclass[ind]
#                         parents.append(parent_label_num_multi)
            
            child_multiclass_ind = row['label_num_multi']
            if child_multiclass_ind not in child_to_parent_map and parents.shape[0] > 0:
                child_to_parent_map[child_multiclass_ind] = parents

        self.child_to_parent_map = child_to_parent_map
        
        ## Dump to pickle
        with open(self.child_to_parent_map_path, 'wb') as file:
            pickle.dump(self.child_to_parent_map, file, protocol=pickle.HIGHEST_PROTOCOL)

#         print(self.child_to_parent_map)
        
        
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
    
    
    def get_data_splits3(self, df, split=(0.7, 0.2, 0.1)):
        """
        Splits according to multiclass label to the split percentages as best as possible
        Unlike get_data_splits2, we DO NOT split according to pre-defined NIH splits
        We simply split each class up according to the percentages
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

    
    
    def get_data_splits2(self, df, split=(0.7, 0.2, 0.1)):
        """
        Splits according to multiclass label to the split percentages as best as possible
        We first split according to pre-defined NIH splits (the ones used when pretraining CheXNet)
        Then we split according to the percentages, the non-nih data per combo label
        """
        
        # Load datasplit if it exists
        if os.path.isfile(self.datasplit_path): 
            with open(self.datasplit_path, 'rb') as file:
                data_splits = pickle.load(file)
        
            return data_splits
        
        # Datasplit does not exist
        
        data_splits = []
        ## Deal with NIH datasplit first
        nih_dataframes = []
        nih_df_sizes = []
        
        df_nih_train = df[df['dataset'] == 'train']
        df_nih_val = df[df['dataset'] == 'val']
        
        nih_df_sizes.extend([len(df_nih_train), len(df_nih_val)])
        nih_dataframes.extend([df_nih_train, df_nih_val])
        
        df_nih_test = df[df['dataset'] == 'test']
        nih_df_sizes.append(len(df_nih_test))
        nih_dataframes.append(df_nih_test)
        
        nih_data_dict = {'label': [], 'count': []}
        df_nih = df[~df['dataset'].isna()]
        image_paths, multiclass_labels = df_nih['image_path'].values, df_nih['label_num_multi'].values
        
        for i in range(df['label_num_multi'].max() + 1):
            rows_with_label = np.where(multiclass_labels == i)
            images_with_label = image_paths[rows_with_label]
            
            df_subsplit = df_nih.loc[df['image_path'].isin(images_with_label)].reset_index(drop=True)

            if len(df_subsplit) > 0:
                label = df_subsplit['label_str'].drop_duplicates().values[0]
                nih_data_dict['label'].append(label)
                nih_data_dict['count'].append(len(df_subsplit))
        
        nih_data_counts = pd.DataFrame.from_dict(nih_data_dict)
        nih_data_counts.to_csv(os.path.join(PATH_TO_DATA_FOLDER, 'nih_data_counts.csv'), index=False)
        print(f'nih_data_counts: \n {nih_data_counts}')
        
        
        # ## Non-nih data
        
        ## Split the rest of the data relatively evenly according to the ratio per class
        ## That is, for each label, the first 70% goes to train, the next 20% to val, the final 10% to test
        df_other = df[df['dataset'].isna()]
        df_other = sklearn.utils.shuffle(df_other, random_state=271) # shuffle
        
        df_other_splits = [pd.DataFrame(columns=df.columns)] * 3
        
        image_paths, multiclass_labels = df_other['image_path'].values, df_other['label_num_multi'].values
        
        non_nih_data_dict = {'label': [], 'count': []}
        
        for i in range(df['label_num_multi'].max() + 1):
            rows_with_label = np.where(multiclass_labels == i)
            images_with_label = image_paths[rows_with_label]
            
            df_subsplit = df_other.loc[df['image_path'].isin(images_with_label)].reset_index(drop=True)

            if len(df_subsplit) > 0:
                label = df_subsplit['label_str'].drop_duplicates().values[0]
                non_nih_data_dict['label'].append(label)
                non_nih_data_dict['count'].append(len(df_subsplit))
                
            val_idx = int(len(df_subsplit) * split[0])
            test_idx = val_idx + int(len(df_subsplit) * split[1])
            df_other_splits[0] = df_other_splits[0].append(df_subsplit.head(val_idx))
            df_other_splits[1] = df_other_splits[1].append(df_subsplit[val_idx:test_idx])
            df_other_splits[2] = df_other_splits[2].append(df_subsplit[test_idx:])
        
        non_nih_data_counts = pd.DataFrame.from_dict(non_nih_data_dict)
        non_nih_data_counts.to_csv(os.path.join(PATH_TO_DATA_FOLDER, 'non_nih_data_counts.csv'), index=False)
        print(f'non_nih_data_counts: \n {non_nih_data_counts}')
        
        for i in range(3):
            df_combined = nih_dataframes[i].append(df_other_splits[i])
            data_splits.append(df_combined)
            
        print([len(data_splits[i]) for i in range(len(data_splits))])
        
        ## Dump to pickle
        with open(self.datasplit_path, 'wb') as file:
            pickle.dump(data_splits, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        return data_splits

    
    
    def get_multiclass_generator_splits(self, df, split=(0.7, 0.3), shuffle_train=True, baseline=False):
        """
        Splitting with tensorflow sequence instead of dataset
        
        Note: split is a 2-length tuple iff baseline == False; otherwise, 3-length
        """
        
#         data_splits = self.get_data_splits2(df, split=(split[0] // 2, split[0] - split[0] // 2, split[1])) ## (train, val, test)
        data_splits = self.get_data_splits3(df, split=(split[0] // 2, split[0] - split[0] // 2, split[1])) ## (train, val, test)
        
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
    
    
    def get_multitask_generator_splits(self, df, split=(0.7, 0.2, 0.1), shuffle_train=True):
        
#         data_splits = self.get_data_splits2(df, split)
        data_splits = self.get_data_splits3(df, split)
        
        # -------------------------------------------------
        df_combined = data_splits[0]
        df_multitask_labels = np.array(df_combined['label_multitask'].to_list())
        df_multitask_labels_sum = np.sum(df_multitask_labels, axis=0)
        untrained_classes = np.where(df_multitask_labels_sum == 0)
        print(f'{untrained_classes[0].shape[0]} classes not trained on: {untrained_classes}')

        df_combined = data_splits[1]
        df_multitask_labels = np.array(df_combined['label_multitask'].to_list())
        df_multitask_labels_sum = np.sum(df_multitask_labels, axis=0)
        unvalidated_classes = np.where(df_multitask_labels_sum == 0)
        print(f'{unvalidated_classes[0].shape[0]} classes not validated on: {unvalidated_classes}')

        df_combined = data_splits[2]
        df_multitask_labels = np.array(df_combined['label_multitask'].to_list())
        df_multitask_labels_sum = np.sum(df_multitask_labels, axis=0)
        untested_classes = np.where(df_multitask_labels_sum == 0)
        print(f'{untested_classes[0].shape[0]} classes not tested on: {untested_classes}')
        # -------------------------------------------------
        
        ## Create image sequences
        full_datasets = []
        for i, ds_type in enumerate(['train', 'val', 'test']):
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
                               batch_size=self.batch_size, num_classes=self.num_classes_multitask)
            full_datasets.append(ds)

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
    
    
    def get_class_weights(self, one_cap=False):
        """ Compute class weighting values for dataset (both individual and combo labels computed)."""
        _, _, indiv, combo = self.get_data_stats(self.df_condensed)
        if one_cap: # Restrains between 0 and 1
            indiv_weights = (combo['count'].sum() - indiv['count'])/combo['count'].sum()
            indiv_weights_false = indiv['count']/combo['count'].sum()
            combo_weights = (combo['count'].sum() - combo['count'])/combo['count'].sum()
            combo_weights_false = combo['count']/combo['count'].sum()
        else: # Unconstrained
            # weight we want to apply if class is True
            indiv_weights = (1 / indiv['count']) * (indiv['count'].sum() / indiv.shape[0]) 
            # weight we want to apply if class is False
            indiv_weights_false = (1 / (indiv['count'].sum()-indiv['count'])) * (indiv['count'].sum() / indiv.shape[0]) 
            combo_weights = (1 / combo['count']) * (combo['count'].sum() / combo.shape[0])
            combo_weights_false = (1 / (combo['count'].sum()-combo['count'])) * (combo['count'].sum() / combo.shape[0]) 
        
        indiv_weights = indiv_weights.sort_index()
        indiv_weights = indiv_weights.drop(['No Finding'])
        indiv_weights_false = indiv_weights_false.sort_index()
        indiv_weights_false = indiv_weights_false.drop(['No Finding'])
        combo_weights = combo_weights.sort_index()
        combo_weights_false = combo_weights_false.sort_index()
        
#         indiv_class_weights = dict(list(enumerate((indiv_weights.values, indiv_weights_false.values))))
#         combo_class_weights = dict(list(enumerate((combo_weights.values, combo_weights_false.values))))
        
#         indiv_class_weights = {}
#         for i in range(len(indiv_weights)):
#             indiv_class_weights = {i: {0: indiv_weights.values[i], 1: indiv_weights_false.values[i]}}
        
        return np.array([indiv_weights.values, indiv_weights_false.values]), np.array([combo_weights.values, combo_weights_false.values])
    


if __name__ == '__main__':
    dataset = MetaChexDataset()
    train_ds = dataset.train_ds
    val_ds = dataset.val_ds
    test_ds = dataset.test_ds

    # Grab one sample
    next(iter(train_ds))
