#         if child_to_parent_map is not None: ## includes childParent
#             self.embedding_matrix = np.load(parent_emb_path) ## [27 x 128]
        
#         self.parent_weight = parent_weight
#         self.child_weight = child_weight
#         self.stage2_weight = stage2_weight
        
#         ## each entry corresponds to multiclass label for that multitask index
#         ## parents who do not exist individually will be marked by a -1 in the self.parent_multiclass_labels array
#         self.parent_multiclass_labels = np.load(parent_multiclass_labels_path) ## (27, ) 
        
#         if child_to_parent_map is not None:
#             self.child_indices = child_to_parent_map.keys()

        
#         self.child_to_parent_map = child_to_parent_map 
#         self.stage_num = stage_num
    
    def supcon_class_loss_proto(self, labels, features):
        
        support_labels = labels[:self.num_classes * self.num_samples_per_class, 1]
        query_labels = labels[self.num_classes * self.num_samples_per_class: self.num_classes * self.num_samples_per_class + self.num_query:, 1]
        labels = tf.concat([support_labels, query_labels], 0)

        return self.class_contrastive_loss(labels, features, proto=True)

    
    def class_contrastive_loss(self, labels, features, proto=False):
        """
        proto: True iff in protonet framework 
        features (ie the z's): [batch_size, embedding_dim]
        labels (ie, the y's): [batch_size, num_labels], where labels are one-hot encoded iff proto=False; otherwise cat
        Assumes self.embedding_matrix is a 28x128 matrix of embedding vectors, where the ith
        column vector is the embedding of parent with label i.
        
        losses.shape (batch_size, )
        """
        losses = np.zeros(features.shape[0])
        weight = self.parent_weight
        child_weight = self.child_weight
        if self.stage_num == 2:
            # For each example in labels, find index where example[index] == 1
            if proto:
                class_labels = labels
            else:
                class_labels = np.where(labels == 1)[1]
            for i, multiclass_label in enumerate(class_labels):
                ## Note: will never encounter parents that do not exist individually (because not in dataset)
                if multiclass_label in self.parent_multiclass_labels: # Parent label (multiclass) and 'no finding' label
                    ## Get corresponding multitask label
                    multitask_label = np.where(self.parent_multiclass_labels == multiclass_label)[0][0]
                    # Update embedding dict with weighted average of existing embedding and mean batch embedding for label
                    self.embedding_matrix[multitask_label] = weight*self.embedding_matrix[multitask_label] + \
                        (1-weight)*tf.reduce_mean(features[np.where(class_labels==multiclass_label)], axis=0)
                    #losses.append(np.zeros(labels.shape[1])) # No childParent loss for parent
                else: # If child, compute loss with average parent embedding as stored in self.embedding_matrix
                    parent_indices = self.child_to_parent_map[multiclass_label]
                    avg_parent_embeds = tf.reduce_mean(self.embedding_matrix[parent_indices], axis=0)
                    
                    ## Update parent embeddings if parent does not exist by itself
                    parents_no_indiv = parent_indices[self.parent_multiclass_labels[parent_indices] == -1]
                    
                    self.embedding_matrix[parents_no_indiv] = child_weight * self.embedding_matrix[parents_no_indiv] + \
                                                              (1 - child_weight) * features[i]
                    
                    losses[i] = tf.reduce_mean(tf.math.square(avg_parent_embeds - features[i])) # squared loss
            
            losses = tf.convert_to_tensor(losses)
        return tf.zeros(self.batch_size)

        
    def supcon_full_loss(self, proto=False):
        """
        features (ie the z's): [batch_size, embedding_dim]
        labels (ie, the y's): [batch_size, num_labels], where labels are one-hot encoded
        """

        def supcon_class_loss_inner(labels, features):
            loss = self.stage2_weight * self.class_contrastive_loss(labels, features)
            return loss

        def supcon_full_loss_inner(labels, features):
            return supcon_label_loss_inner(labels, features) + supcon_class_loss_inner(labels, features)
        
        def proto_supcon_full_loss_inner(labels, features):
            """
            labels: [n + n_query, 2]; proto-labels: labels[:, 0]; multiclass_labels: labels[:, 1]
            features: [n * k + n_query, 128]
            """
            
            return self.supcon_label_loss_proto(labels, features) + self.supcon_class_loss_proto(labels, features)

        
        return proto_supcon_full_loss_inner if proto else supcon_full_loss_inner
    
    
        def proto_and_supcon_loss(self):
        def proto_and_supcon_loss_inner(labels, features):
            supcon_loss = self.supcon_label_loss_proto(labels, features)[self.num_classes * self.num_samples_per_class: self.num_classes * self.num_samples_per_class + self.num_query]
            proto_loss = self.proto_loss_inner(labels, features)
            
#             print(f"supcon_loss.shape: {supcon_loss.shape} \t proto_loss.shape: {proto_loss.shape}")
            return supcon_loss + proto_loss
        
        return proto_and_supcon_loss_inner
    