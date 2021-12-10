"""
All the methods that are unused from dataloader
"""

def get_class_probs(self):
    """ Compute class probabilities for dataset (both individual and combo labels computed)"""
    _, _, indiv, combo = self.get_data_stats(self.df_condensed)
    indiv_class_probs = indiv['count']/indiv['count'].sum()
    combo_class_probs = combo['count']/combo['count'].sum()

    return indiv_class_probs, combo_class_probs

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
    self.steps_per_epoch = (len(nih_datasets[0]) / self.batch_size) * 0.1
    #return nih_datasets ## early return for now


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


def shuffle_and_batch(self, ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=100)
    ds = ds.map(self.load_and_preprocess_image) ## maps the preprocessing step
    ds = ds.batch(self.batch_size)
#         ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


"""Comments from get_generator_splits()"""
        ## To see which labels are represented in training set
#         df_combined = nih_dataframes[0].append(df_other_splits[0])
#         df_multitask_labels = np.array(df_combined['label_multitask'].to_list())
#         df_multitask_labels_sum = np.sum(df_multitask_labels, axis=0)
#         untrained_classes = np.where(df_multitask_labels_sum == 0)
#         print(df_multitask_labels_sum)
#         print(f'classes not trained on: {untrained_classes}')
        
#         df_combined = nih_dataframes[1].append(df_other_splits[1])
#         df_multitask_labels = np.array(df_combined['label_multitask'].to_list())
#         df_multitask_labels_sum = np.sum(df_multitask_labels, axis=0)
#         unvalidated_classes = np.where(df_multitask_labels_sum == 0)
#         print(df_multitask_labels_sum)
#         print(f'classes not validated on: {unvalidated_classes}')
        
#         df_combined = nih_dataframes[2].append(df_other_splits[2])
#         df_multitask_labels = np.array(df_combined['label_multitask'].to_list())
#         df_multitask_labels_sum = np.sum(df_multitask_labels, axis=0)
#         untested_classes = np.where(df_multitask_labels_sum == 0)
#         print(df_multitask_labels_sum)
#         print(f'classes not tested on: {untested_classes}')
        
        
#         df_multitask_labels = np.array(df['label_multitask'].to_list())
#         df_multitask_labels_sum = np.sum(df_multitask_labels, axis=0)
#         print(df_multitask_labels_sum)
#         classes_that_need_to_be_augmented = np.where(df_multitask_labels_sum < 100)
#         print(df_multitask_labels_sum)
#         print(f'classes that need to be augmented: {classes_that_need_to_be_augmented}')
        
        
    
        #########
        
    def get_data_splits(self, df, split=(0.8, 0.1, 0.1)):
        """Splitting with tensorflow sequence instead of dataset"""
        
        # Load datasplit if it exists
        if os.path.isfile(self.datasplit_path): 
            with open(self.datasplit_path, 'rb') as file:
                data_splits = pickle.load(file)
        else:
            ## Deal with NIH datasplit first
            nih_dataframes = []
            nih_df_sizes = []

            for ds_type in ['train', 'val', 'test']:
                df_nih = df[df['dataset'] == ds_type]
                nih_df_sizes.append(len(df_nih))
                nih_dataframes.append(df_nih)

            ## Non-nih data

            ## Split the rest of the data relatively evenly according to the ratio per class
            ## That is, for each label, the first 80% goes to train, the next 10% to val, the final 10% to test
            df_other = df[df['dataset'].isna()]
            df_other = sklearn.utils.shuffle(df_other, random_state=271) # shuffle

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
        
            data_splits = []
            for i, ds_type in enumerate(['train', 'val', 'test']):
                df_combined = nih_dataframes[i].append(df_other_splits[i])
                df_combined = sklearn.utils.shuffle(df_combined, random_state=271) # shuffle
                data_splits.append(df_combined)
            
            ## Dump to pickle
            with open(self.datasplit_path, 'wb') as file:
                pickle.dump(data_splits, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        print([len(data_splits[i]) for i in range(3)])
        return data_splits
    
        
class ChexNetWithSupCon():
    
    def __init__(self, encoder_with_proj_head,
                 stage1_optim, dataset, supcon_loss_weights,
                 stage1_num_epochs):
        
        """
        encoder_with_proj_head: The encoder CNN (pre-trained CheXNet) with projection head (FC layer + norm)
        stage1_optim: optimizer
        dataset: MetaChex generator
        supcon_loss_weights: weights for the supcon loss terms
        """
        
        self.model = encoder_with_proj_head
        
        
        
    def stage1_train_step(self, model, x, y):
        """
        model: The encoder CNN (pre-trained CheXNet) + the projection head (FC layer + norm)
        x: inputs [batch_size, 3, 224, 224] (not sure if the channel dimension is in the correct place)
        y: labels [batch_size, num_labels] (one-hot encoded)
        """

        with tf.GradientTape() as tape:
            z = model(x) ## [batch_size, embedding_dim]
            z = z.reshape([z.shape[0], 1, -1]) ## [batch_size, 1, embedding_dim]

            supcon_label_loss = Losses().supcon_label_loss(z, y)
            supcon_class_loss = Losses().supcon_class_loss(z, y)

        supcon_total_loss = self.supcon_loss_weights[0] * supcon_label_loss + \
                            self.supcon_loss_weights[1] * supcon_class_loss

        gradients = tape.gradient(supcon_total_loss, model.trainable_variables)
        optim.apply_gradients(zip(gradients, model.trainable_variables))

        return supcon_total_loss, supcon_label_loss, supcon_class_loss


    def stage1_eval(self, model, x, y):
        """
        model: The encoder CNN (pre-trained CheXNet) + the projection head (FC layer + norm)
        x: inputs [batch_size, 3, 224, 224] (not sure if the channel dimension is in the correct place)
        y: labels [batch_size, num_labels] (one-hot encoded)
        """

        z = model(x) ## [batch_size, embedding_dim]
        z = z.reshape([z.shape[0], 1, -1]) ## [batch_size, 1, embedding_dim]

        supcon_label_loss = Losses().supcon_label_loss(z, y)
        supcon_class_loss = Losses().supcon_class_loss(z, y)

        supcon_total_loss = self.supcon_loss_weights[0] * supcon_label_loss + \
                            self.supcon_loss_weights[1] * supcon_class_loss

        return supcon_total_loss, supcon_label_loss, supcon_class_loss


    def stage1_training(self):
        """
        Training the encoder
        Returns list of the validation supcon losses
        """

        self.encoder.trainable = True
        self.projection_head.trainable = True

        model = tf.keras.Sequential([self.encoder, self.projection_head,])

        val_supcon_loss_logs = [[], [], []]

        for ep in range(self.stage1_num_epochs):
            num_steps = self.dataset.train_ds.steps
            for step in range(num_steps):
                batch_x, batch_y = self.dataset.train_ds[step]
                supcon_total_loss, supcon_label_loss, supcon_class_loss = self.stage1_train_step(model, batch_x, batch_y)

                print('[epoch {}/{}, train_step {}/{}] => supcon_total_loss: {:.5f}, supcon_label_loss: {:.5f}, \
                      supcon_class_loss: {:.5f}'.format(ep+1, self.stage1_num_epochs, step+1, num_steps, supcon_total_loss,
                                                        supcon_label_loss, supcon_class_loss)


            ## Validation loss
            val_total_loss, val_label_loss, val_class_loss = 0, 0, 0
            num_val_steps = self.dataset.val_ds.steps
            for step in range(num_val_steps):
                batch_x, batch_y = self.dataset.val_ds[step]
                val_total_loss, val_label_loss, val_class_loss += self.stage1_eval(model, batch_x, batch_y)

                print('[epoch {}/{}, val_step {}/{}] => val_supcon_total_loss: {:.5f}, val_supcon_label_loss: {:.5f}, \
                      val_supcon_class_loss: {:.5f}'.format(ep+1, self.stage1_num_epochs, step+1, num_val_steps, val_total_loss,
                                                        val_label_loss, val_class_loss)

            val_supcon_loss_logs[0].append(val_total_loss / num_val_steps)
            val_supcon_loss_logs[1].append(val_label_loss / num_val_steps)
            val_supcon_loss_logs[2].append(val_class_loss / num_val_steps)

            print('[epoch {}/{}] => avg_val_supcon_total_loss: {:.5f}, avg_val_supcon_label_loss: {:.5f}, \
                      avg_val_supcon_class_loss: {:.5f}'.format(ep+1, self.stage1_num_epochs, val_supcon_loss_logs[0][-1],
                                                           val_supcon_loss_logs[1][-1], val_supcon_loss_logs[2][-1])          

        return val_supcon_loss_log

    def get_supcon_stage1_ds(self):
        """
        Get dataset for stage1 training (indiv parents and children of parents that don't exist individually)
        in training set
        """
        stage1_df = pd.DataFrame(columns=self.train_df.columns)
        
        label_multitask_arr = np.array(self.train_df['label_multitask'].to_list()) ## [len(train_df), 27]
        row_indices, multitask_indices = np.where(label_multitask_arr == 1)

        for i, label in enumerate(self.parent_multiclass_labels):
            if label != -1: ## Sample parents that exist individually
                df_class = self.train_df[self.train_df['label_num_multi'] == label]

            else: ## get children of parents that don't exist individually
                ## Get rows where multitask_indices includes i
                children_rows = row_indices[multitask_indices == i]
                df_class = train_df.iloc[children_rows]

            df_class['parent_id'] = i ## label with parent class
            stage1_df = stage1_df.append(df_class)

        stage1_df = stage1_df.reset_index(drop=True)
        steps = int(len(stage1_df) / self.batch_size * 0.1)
        
        stage1_ds = ImageSequence(df=stage1_df, steps=steps, shuffle_on_epoch_end=True, parents_only=True,
                               num_classes=self.num_classes_multiclass, multiclass=True, batch_size=self.batch_size)
    
        return stage1_ds
        
        
        ## Extract rows of indiv parents
#         label_num_multi_arr = self.train_df['label_num_multi'].values
#         mask = np.in1d(label_num_multi_arr, self.parent_multiclass_labels)
#         indices = np.arange(mask.shape[0])[mask]
#         indiv_parent_rows = self.train_df.iloc[indices]
        
#         ## Extract children of parents that don't exist individually
#         combo_parent_mask = ~np.in1d(self.parent_multiclass_labels, np.unique(indiv_parent_rows['label_num_multi'].values))
#         combo_parent_indices = np.arange(combo_parent_mask.shape[0])[combo_parent_mask]
# #         print(combo_parent_indices)
        
#         label_multitask_arr = np.array(self.train_df['label_multitask'].to_list()) 
#         row_indices, multitask_indices = np.where(label_multitask_arr == 1)
        
#         children_row_indices = []
#         for i, row in self.train_df.iterrows():
#             ## get rows where there is a non-empty intersection between combo_parent_indices and row multitask indices
#             indices_for_row = np.where(row_indices == i)
#             multitask_indices_for_row = multitask_indices[indices_for_row]
            
#             inter = np.intersect1d(combo_parent_indices, multitask_indices_for_row)
#             if inter.shape[0] > 0: 
#                 children_row_indices.append(i)

#         children_rows = self.train_df.iloc[children_row_indices]
        
#         stage1_df = indiv_parent_rows.append(children_rows).reset_index(drop=True)
#         steps = int(len(stage1_df) / self.batch_size * 0.1)
        
#         stage1_ds = ImageSequence(df=stage1_df, steps=steps, shuffle_on_epoch_end=True, 
#                                num_classes=self.num_classes_multiclass, multiclass=True, batch_size=self.batch_size)
    
#         return stage1_ds
                  
                  
def get_protonet_generator_splits(self, df, n, k, n_query, n_test, k_test, n_test_query, 
                                      split=(0.7, 0.1, 0.2), shuffle_train=True):
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
            
        total = df_combo_nums['count'].values.sum()
        print(f'Total number of images: {total}')
        df_combo_counts.to_csv('data/df_combo_counts.csv', index=False)
        df_label_nums.to_csv('data/df_label_nums.csv', index=True)
        df_combo_nums.to_csv('data/df_combo_nums.csv', index=True)

        return unique_labels_dict, df_combo_counts, df_label_nums, df_combo_nums
                  
                  
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