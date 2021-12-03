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