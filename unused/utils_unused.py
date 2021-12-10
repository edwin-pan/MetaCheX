def load_chexnet_pretrained(class_names=np.arange(14), weights_path='chexnet_weights.h5', 
                            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)):

    img_input = tf.keras.layers.Input(shape=input_shape)
    base_model = tf.keras.applications.densenet.DenseNet121(include_top=False, #weights=None,  # use imagenet weights
                                                            input_tensor=img_input, pooling='avg')
    base_model.trainable = False


    x = base_model.output
    predictions = tf.keras.layers.Dense(len(class_names), activation="sigmoid", name="predictions")(x)
    model = tf.keras.models.Model(inputs=img_input, outputs=predictions)

    return model


def get_proto_acc_for_class(num_classes, num_samples_per_class, num_query, labels, features, class_idx):
    prototypes, queries, query_labels = extract_prototypes_and_queries(num_classes, num_samples_per_class,
                                                                                         num_query, labels, features)
            
    query_preds = get_nearest_neighbour(queries, prototypes)

    try:
        mask = tf.cast(labels[-num_query:, class_idx], tf.bool) 
        num = np.sum(mask)
        if num == 0:
            raise ZeroDivisionError
        query_preds = query_preds[np.array(mask)]
        query_labels = query_labels[np.array(mask)]
        num_correct = np.where(query_preds == query_labels)[0].shape[0]
        acc = num_correct / num
    except ZeroDivisionError:
        acc = tf.convert_to_tensor([])

    return acc

def proto_acc_covid_outer(num_classes=5, num_samples_per_class=3, num_query=5):
    def proto_acc_covid(labels, features):
        """
        labels: [n * k + n_query + n_query, 2] 
           - for labels[:n x k + n_query] -- proto_labels: labels[:, 0]; multiclass_labels: labels[:, 1]
           - for labels[-n_query: ] -- covid mask: labels[:, 0]; tb_mask: labels[:, 1]
        features: [n * k + n_query, 128]
        """
        covid_acc = get_proto_acc_for_class(num_classes, num_samples_per_class, num_query, labels, features, class_idx=0)

        return covid_acc
    return proto_acc_covid

def proto_acc_tb_outer(num_classes=5, num_samples_per_class=3, num_query=5):
    def proto_acc_tb(labels, features):
        """
        labels: [n * k + n_query + n_query, 2] 
           - for labels[:n x k + n_query] -- proto_labels: labels[:, 0]; multiclass_labels: labels[:, 1]
           - for labels[-n_query: ] -- covid mask: labels[:, 0]; tb_mask: labels[:, 1]
        features: [n * k + n_query, 128]
        """
        tb_acc = get_proto_acc_for_class(num_classes, num_samples_per_class, num_query, labels, features, class_idx=1)

        return tb_acc
    
    return proto_acc_tb

def proto_acc_covid_tb_outer(num_classes=5, num_samples_per_class=3, num_query=5):
    def proto_acc_covid_tb(labels, features):
        """
        labels: [n * k + n_query + n_query, 2] 
           - for labels[:n x k + n_query] -- proto_labels: labels[:, 0]; multiclass_labels: labels[:, 1]
           - for labels[-n_query: ] -- covid mask: labels[:, 0]; tb_mask: labels[:, 1]
        features: [n * k + n_query, 128]
        """
        prototypes, queries, query_labels = extract_prototypes_and_queries(num_classes, num_samples_per_class,
                                                                                         num_query, labels, features)
            
        query_preds = get_nearest_neighbour(queries, prototypes)
        
        ## COVID|TB acc
        try:
            covid_tb_mask = tf.cast(labels[-num_query:, 0], tf.bool) | tf.cast(labels[-num_query:, 1], tf.bool)
            num_covid_tb = np.sum(covid_tb_mask)
            if num_covid_tb == 0:
                raise ZeroDivisionError
            covid_tb_query_preds = query_preds[np.array(covid_tb_mask)]
            covid_tb_query_labels = query_labels[np.array(covid_tb_mask)]
            num_correct = np.where(covid_tb_query_preds == covid_tb_query_labels)[0].shape[0]
            covid_tb_acc = num_correct / num_covid_tb
        except ZeroDivisionError:
#             print("no covid or tb in meta-test task")
            covid_tb_acc = tf.convert_to_tensor([])
        
        return covid_tb_acc
    return proto_acc_covid_tb

def average_precision(y_true, y_pred, dataset, dir_path=".", plot=True):
    
    test_ap_log_path = os.path.join(dir_path, "average_prec.txt")
    with open(test_ap_log_path, "w") as f:
        aps = []
        for i in range(y_true.shape[1]):
            try:
                ap = average_precision_score(y_true[:, i], y_pred[:, i])
                if plot:
                    pr_plot_dir = os.path.join(dir_path, 'pr_plots')
                    os.makedirs(pr_plot_dir, exist_ok=True)
                    precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
                    display = PrecisionRecallDisplay(recall=recall, precision=precision, average_precision=ap)
                    display.plot()
                    plot = display.figure_
                    plot.savefig(os.path.join(pr_plot_dir, f"{dataset.unique_labels[i]}_pr_curve.png"))
                    plt.close()
                aps.append(ap)
            except RuntimeWarning:
                print(f'{dataset.unique_labels[i]} not tested on')
                ap = 'N/A'
            f.write(f"{dataset.unique_labels[i]}: {ap}\n")
        mean_ap = np.mean(aps)
        f.write("-------------------------\n")
        f.write(f"mean average precision: {mean_ap}\n")
    print(f"mean average precision: {mean_ap}")
    
    
    def get_sampled_ds(ds, multiclass=True, max_per_class=20):
    
    num_classes = ds.num_classes
    if multiclass:
        sampled_df = get_sampled_df_multiclass(ds.df, num_classes=num_classes, max_per_class=max_per_class)
    else:
        sampled_df = get_sampled_df_multitask(ds.df, num_classes=num_classes, max_per_class=max_per_class)
    
    sampled_ds = ImageSequence(sampled_df, shuffle_on_epoch_end=False, num_classes=num_classes, multiclass=multiclass)
    
    return sampled_ds


def get_sampled_df_multitask(train_df, num_classes, max_per_class=20):
    """
    Sample max_per_class samples from each (multitask) class in train_df -- repeats are ok
    """
    sampled_df = pd.DataFrame(columns=train_df.columns)
    
    label_multitask_arr = np.array(train_df['label_multitask'].to_list()) ## [len(train_df), 27]
    row_indices, multitask_indices = np.where(label_multitask_arr == 1)

    for i in range(num_classes):
        children_rows = row_indices[multitask_indices == i]
        df_class = train_df.iloc[children_rows]

        if len(df_class) > max_per_class:
            df_class = df_class.sample(max_per_class)

        sampled_df = sampled_df.append(df_class)
    
    sampled_df = sampled_df.reset_index(drop=True)
    return sampled_df

def get_sampled_df_multiclass(train_df, num_classes, parents_only=False, max_per_class=20):
    """
    Sample max_per_class samples from each class in train_df
    if self.parents_only, sample only the parents that exist and the children of parents that don't
    """
    sampled_df = pd.DataFrame(columns=train_df.columns)

    if not parents_only:
        for i in range(num_classes):
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