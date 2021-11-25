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

from tensorflow.keras.utils import Sequence
from PIL import Image
import sklearn
from skimage.transform import resize

np.random.seed(271)


class ImageSequence(Sequence):
    """
    Class was taken from CheXNet implementation
    """

    def __init__(self, df, batch_size=8,
                 target_size=(224, 224), augmenter=None, verbose=0, steps=None,
                 shuffle_on_epoch_end=True, random_state=271, num_classes=27, multiclass=False):
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
        self.augmenter = augmenter ## not used
        self.verbose = verbose
        self.shuffle = shuffle_on_epoch_end
        self.random_state = random_state
        self.num_classes = num_classes
        self.multiclass = multiclass
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
        batch_x_path = self.x_path[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.asarray([self.load_image(x_path) for x_path in batch_x_path])
        # batch_x = self.transform_batch_images(batch_x)
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

    def load_image(self, image_path):
        image = Image.open(image_path)
        image_array = np.asarray(image.convert("RGB"))
        image_array = image_array / 255.
        image_array = resize(image_array, self.target_size)
        return image_array

    def transform_batch_images(self, batch_x):
        if self.augmenter is not None:
            batch_x = self.augmenter.augment_images(batch_x)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        batch_x = (batch_x - imagenet_mean) / imagenet_std
        batch_x = tf.convert_to_tensor(batch_x)
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
        return self.y[:self.steps*self.batch_size, :]

    def prepare_dataset(self):
        df = self.df.sample(frac=1., random_state=self.random_state)
        self.x_path = df['image_path']
        if self.multiclass:
            self.y = np.eye(int(self.num_classes))[df['label_num_multi'].values.astype(int)]
        else:
            self.y = np.array(df['label_multitask'].to_list())
        print("self.y.shape: ", self.y.shape)

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()


class MetaChexDataset():
    def __init__(self, shuffle_train=True, multiclass=False, batch_size=8):
        self.batch_size = batch_size
        self.multiclass = multiclass
        
        # Reads in data.pkl file (mapping of paths to unformatted labels)
        self.data_path = os.path.join(PATH_TO_DATA_FOLDER, 'data.pkl')
        
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
        
        if multiclass:
            print("[INFO] get child-to-parents mapping")
            ## list of multiclass labels corresponding to multitask index
            self.parent_multiclass_labels = np.ones((self.num_classes_multitask,)) * -1 
            self.get_child_to_parent_mapping()
            self.get_num_parents_to_count_df()
            self.num_classes_multiclass_plus = np.max(self.parent_multiclass_labels)

        print("[INFO] constructing tf train/val/test vars")
         ## already shuffled and batched
        print("[INFO] shuffle & batch")
        if multiclass:
            [self.train_ds, self.test_ds] = self.get_multiclass_generator_splits(self.df_condensed, shuffle_train=shuffle_train)
            self.get_supcon_stage1_ds()
        else:
            [self.train_ds, self.val_ds, self.test_ds] = self.get_multitask_generator_splits(self.df_condensed, shuffle_train=shuffle_train)

        print('[INFO] initialized')
        return

    
    def get_supcon_stage1_ds(self):
        """
        Get dataset for stage1 training (indiv parents and children of parents that don't exist individually)
        in training set
        """
        ## Extract rows of indiv parents
        label_num_multi_arr = self.train_df['label_num_multi'].values
        mask = np.in1d(label_num_multi_arr, self.parent_multiclass_labels)
        indices = np.arange(mask.shape[0])[mask]
        indiv_parent_rows = self.train_df.iloc[indices]
        
        ## Extract children of parents that don't exist individually
        combo_parent_mask = ~np.in1d(self.parent_multiclass_labels, np.unique(indiv_parent_rows['label_num_multi'].values))
        combo_parent_indices = np.arange(combo_parent_mask.shape[0])[combo_parent_mask]
#         print(combo_parent_indices)
        
        label_multitask_arr = np.array(self.train_df['label_multitask'].to_list()) 
        row_indices, multitask_indices = np.where(label_multitask_arr == 1)
        
        children_row_indices = []
        for i, row in self.train_df.iterrows():
            ## get rows where there is a non-empty intersection between combo_parent_indices and row multitask indices
            indices_for_row = np.where(row_indices == i)
            multitask_indices_for_row = multitask_indices[indices_for_row]
            
            inter = np.intersect1d(combo_parent_indices, multitask_indices_for_row)
            if inter.shape[0] > 0: 
                children_row_indices.append(i)

        children_rows = self.train_df.iloc[children_row_indices]
        
        stage1_df = indiv_parent_rows.append(children_rows).reset_index(drop=True)
        steps = int(len(stage1_df) / self.batch_size * 0.1)
        
        self.stage1_ds = ImageSequence(df=stage1_df, steps=steps, shuffle_on_epoch_end=True, 
                               num_classes=self.num_classes_multiclass_plus, multiclass=True, batch_size=self.batch_size)
    
        
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
        
        if os.path.isfile(self.child_to_parent_map_path): ## load from pickle
            with open(self.child_to_parent_map_path, 'rb') as file:
                self.child_to_parent_map = pickle.load(file)
            
            ## Load from pickle
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

            ## This chunk is not necessary (gets multitask to multiclass mapping for parents that exist individually)
            if multitask_indices_for_row.shape[0] == 1: # indiv class
                parent_label_num_multi = row['label_num_multi']
                if self.parent_multiclass_labels[multitask_indices_for_row[0]] == -1:
                    self.parent_multiclass_labels[multitask_indices_for_row[0]] = parent_label_num_multi
        
        
        ## Populate the rest of the self.parent_multiclass_labels (parents that only exist as combos)
        ## I realized that this is not necessary, but just in case we somehow need it
        indiv_parent_indices = np.where(self.parent_multiclass_labels == -1)[0]
        self.parent_multiclass_labels[indiv_parent_indices] = np.arange(indiv_parent_indices.shape[0]) + self.num_classes_multiclass
        
        print(self.parent_multiclass_labels)

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

        print(self.child_to_parent_map)
        
        
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
    
    
    def get_multiclass_generator_splits(self, df, split=(0.8, 0.2), shuffle_train=True):
        """Splitting with tensorflow sequence instead of dataset"""
        
#         print(df[df['label_num_multi'] == 322]['label_str'])
#         exit(1)
        
        ## Deal with NIH datasplit first
        nih_dataframes = []
        nih_df_sizes = []
        
        df_nih_train = df[df['dataset'] == 'train']
        df_nih_val = df[df['dataset'] == 'val']
        nih_df_sizes.append(len(df_nih_train) + len(df_nih_val))
        nih_dataframes.append(df_nih_train.append(df_nih_val).reset_index(drop=True))
        
        df_nih_test = df[df['dataset'] == 'test']
        nih_df_sizes.append(len(df_nih_test))
        nih_dataframes.append(df_nih_test)
        
        # ## Non-nih data
        
        ## Split the rest of the data relatively evenly according to the ratio per class
        ## That is, for each label, the first 80% goes to train, the next 10% to val, the final 10% to test
        df_other = df[df['dataset'].isna()]
        df_other = sklearn.utils.shuffle(df_other) # shuffle
        
        df_other_splits = [pd.DataFrame(columns=df.columns)] * 2
        
        image_paths, multiclass_labels = df_other['image_path'].values, df_other['label_num_multi'].values
        
        for i in range(df['label_num_multi'].max() + 1):
            rows_with_label = np.where(multiclass_labels == i)
            images_with_label = image_paths[rows_with_label]
            
            df_subsplit = df_other.loc[df['image_path'].isin(images_with_label)].reset_index(drop=True)
            
            if len(df_subsplit) > 0:
                print(i)
            
            test_idx = int(len(df_subsplit) * split[0])
            df_other_splits[0] = df_other_splits[0].append(df_subsplit.head(test_idx))
            df_other_splits[1] = df_other_splits[1].append(df_subsplit[test_idx:])
        
        df_combined = nih_dataframes[0].append(df_other_splits[0])
        untrained_classes = set(range(multiclass_labels.max() + 1)) - set(df_combined['label_num_multi'].values)
        print(f'classes not trained on: {untrained_classes}')
        
        df_combined = nih_dataframes[1].append(df_other_splits[1])
        untested_classes = set(range(multiclass_labels.max() + 1)) - set(df_combined['label_num_multi'].values)
        print(f'classes not tested on: {untested_classes}, num: {len(untested_classes)}')
#         exit(1)
        
        full_datasets = []
        
        for i, ds_type in enumerate(['train', 'test']):
            
            df_combined = nih_dataframes[i].append(df_other_splits[i])
            
            
            if ds_type == 'train':
                shuffle_on_epoch_end = shuffle_train
                factor = 0.1
                steps = int(len(df_combined) / self.batch_size * factor)
                self.train_steps_per_epoch = steps
                
                self.train_df = df_combined.reset_index(drop=True)
            else:
                steps = None
                shuffle_on_epoch_end = False
                
            df_combined = sklearn.utils.shuffle(df_combined) # shuffle
            ds = ImageSequence(df=df_combined, steps=steps, shuffle_on_epoch_end=shuffle_on_epoch_end, 
                               num_classes=self.num_classes_multiclass_plus, multiclass=True, batch_size=self.batch_size)
            full_datasets.append(ds)
        
        return full_datasets
    
    
    def get_multitask_generator_splits(self, df, split=(0.8, 0.1, 0.1), shuffle_train=True):
        """Splitting with tensorflow sequence instead of dataset"""
        
        ## Deal with NIH datasplit first
        nih_dataframes = []
        nih_df_sizes = []
        
        for ds_type in ['train', 'val', 'test']:
            df_nih = df[df['dataset'] == ds_type]
            nih_df_sizes.append(len(df_nih))
            nih_dataframes.append(df_nih)
        
        # ## Non-nih data
        
        ## Split the rest of the data relatively evenly according to the ratio per class
        ## That is, for each label, the first 80% goes to train, the next 10% to val, the final 10% to test
        df_other = df[df['dataset'].isna()]
        df_other = sklearn.utils.shuffle(df_other) # shuffle
        
        df_other_splits = [pd.DataFrame(columns=df.columns)] * 3
        
        image_paths, multitask_labels = df_other['image_path'].values, np.array(df_other['label_multitask'].to_list())
        
        for i in range(multitask_labels.shape[1]):
            rows_with_label = multitask_labels[:, i] == 1
            images_with_label = image_paths[rows_with_label]
            
            df_subsplit = df_other.loc[df['image_path'].isin(images_with_label)].reset_index(drop=True)
            
            val_idx = int(len(df_subsplit) * split[0])
            test_idx = int(len(df_subsplit) * (split[0] + split[1]))
            df_other_splits[0] = df_other_splits[0].append(df_subsplit.head(val_idx))
            df_other_splits[1] = df_other_splits[1].append(df_subsplit[val_idx:test_idx])
            df_other_splits[2] = df_other_splits[2].append(df_subsplit[test_idx:])
        
        df_other_train_val_same = df_other_splits[1].loc[df_other_splits[1]['image_path'].isin(df_other_splits[0]['image_path'])].copy()
        df_other_train_test_same = df_other_splits[2].loc[df_other_splits[2]['image_path'].isin(df_other_splits[0]['image_path'])].copy()
        df_other_val_test_same = df_other_splits[2].loc[df_other_splits[2]['image_path'].isin(df_other_splits[1]['image_path'])].copy()
#         print('train/val overlap: ', len(df_other_train_val_same))
#         print('train/test overlap; ', len(df_other_train_test_same))
#         print('val/test overlap; ', len(df_other_val_test_same))
        
        ## remove train/test and val/test overlaps on train and val sets (keep in the test set -- ensures all labels tested on)
        df_other_splits[0] = df_other_splits[0].loc[~df_other_splits[0]['image_path'].isin(df_other_train_test_same['image_path'])]
        df_other_splits[1] = df_other_splits[1].loc[~df_other_splits[1]['image_path'].isin(df_other_val_test_same['image_path'])]
        
        ## remove train/val overlap on the val set 
        df_other_splits[1] = df_other_splits[1].loc[~df_other_splits[1]['image_path'].isin(df_other_train_val_same['image_path'])]
        
        ## drop duplicates in val and test (duplicates ok in train -- oversampling-ish)
        df_other_splits[1].drop_duplicates(subset='image_path', inplace=True)
        df_other_splits[2].drop_duplicates(subset='image_path', inplace=True)
        
        full_datasets = []
        
        for i, ds_type in enumerate(['train', 'val', 'test']):
            df_combined = nih_dataframes[i].append(df_other_splits[i])
            
            if ds_type == 'train':
                shuffle_on_epoch_end = shuffle_train
                factor = 0.1
            else:
                shuffle_on_epoch_end = False
                factor = 0.2
            
            steps = int(len(df_combined) / self.batch_size * factor)
            
            if ds_type == 'train':
                self.train_steps_per_epoch = steps
            elif ds_type == 'val':
                self.val_steps_per_epoch = steps
                
            df_combined = sklearn.utils.shuffle(df_combined) # shuffle
            
            ds = ImageSequence(df=df_combined, steps=steps, shuffle_on_epoch_end=shuffle_on_epoch_end, 
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
            
            self.unique_labels = list(self.unique_labels_dict.keys())
        
        self.num_classes_multitask = len(self.unique_labels) - 1 ## remove no finding
        self.num_classes_multiclass = max(df['label_num_multi'].values) + 1
        
        if self.multiclass:
            label_df = df[['label_num_multi', 'label_str']].drop_duplicates().sort_values('label_num_multi')
            self.unique_labels = label_df['label_str'].values
            
        return df
    
    
    def get_class_weights(self, one_cap=False):
        """ Compute class weighting values for dataset (both individual and combo labels computed)."""
        _, _, indiv, combo = self.get_data_stats(self.df_condensed)
        if one_cap: # Restrains between 0 and 1
            indiv_weights = (combo['count'].sum() - indiv['count'])/combo['count'].sum()
            indiv_weights_false = indiv['count']/combo['count'].sum()
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
        
        indiv_class_weights = {}
        for i in range(len(indiv_weights)):
            indiv_class_weights = {i: {0: indiv_weights.values[i], 1: indiv_weights_false.values[i]}}
        
        return np.array([indiv_weights.values, indiv_weights_false.values]), indiv_class_weights, combo_class_weights
    

    def get_class_probs(self):
        """ Compute class probabilities for dataset (both individual and combo labels computed)"""
        _, _, indiv, combo = self.get_data_stats(self.df_condensed)
        indiv_class_probs = indiv['count']/indiv['count'].sum()
        combo_class_probs = combo['count']/combo['count'].sum()
        
        return indiv_class_probs, combo_class_probs
    


if __name__ == '__main__':
    dataset = MetaChexDataset()
    train_ds = dataset.train_ds
    val_ds = dataset.val_ds
    test_ds = dataset.test_ds

    # Grab one sample
    next(iter(train_ds))
