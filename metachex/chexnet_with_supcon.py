import numpy as np
import tensorflow as tf
from loss import Losses

class ChexNetWithSupCon():
    
    def __init__(self, encoder, projection_head, prediction_head, 
                 stage1_optim, stage2_optim, dataset, supcon_loss_weights,
                 stage1_num_epochs, stage2_num_epochs, stage2_class_weights):
        
        """
        encoder: The encoder CNN (pre-trained CheXNet)
        projection_head: the projection head (FC layer + norm)
        prediction_head: the prediction head (MLP)
        stage1_optim: optimizer
        stage2_optim: optimizer
        dataset: MetaChex dataset
        supcon_loss_weights: weights for the supcon loss terms
        """
        
        self.encoder = encoder
        self.projection_head = projection_head
        self.prediction_head = prediction_head
        self.stage1_optim = stage1_optim
        self.stage2_optim = stage2_optim
        self.dataset = dataset
        self.supcon_loss_weights = supcon_loss_weights
        self.stage1_num_epochs = stage1_num_epochs
        self.stage2_num_epochs = stage2_num_epochs
        self.stage2_class_weights = stage2_class_weights
        
        
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


    def stage2_training(self):
        """
        Stage 2 (prediction) training
        """

        self.encoder.trainable = False
        self.prediction_head.trainable = True
        model = tf.keras.Sequential([self.encoder, self.prediction_head,])

        checkpoint_path = "supcon_training_progress/cp.ckpt" # path for saving model weights
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

        model.compile(optimizer=self.stage2_optim,
                     loss='crossentropy',
                     metrics=[tf.keras.metrics.AUC(multi_label=True),  'binary_accuracy', 'accuracy', 
                             tfa.metrics.F1Score(average='micro',num_classes=dataset.num_classes_multiclass), 
                             tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
                      run_eagerly=True)

        model.fit(dataset.train_ds,
                validation_data=self.dataset.val_ds,
                epochs=self.stage2_num_epochs,
                steps_per_epoch=self.dataset.train_steps_per_epoch, ## size(train_ds) * 0.125 * 0.1
                validation_steps=self.dataset.val_steps_per_epoch, ## size(val_ds) * 0.125 * 0.2
                batch_size=self.dataset.batch_size, ## 8
                class_weight=self.stage2_class_weights, 
                callbacks=[cp_callback])

