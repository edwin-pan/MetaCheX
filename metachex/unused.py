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
