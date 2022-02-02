import os
import pickle

import fire
import tensorflow as tf
from sklearn.model_selection import train_test_split

# # check if tensorflow sess is working
# with tf.compat.v1.Session() as sess:
#     x = tf.constant('Tensorflow works!')
#     print(sess.run(x))

# # Yale Facial data input (img shape: 50*50)
num_input = 50 * 50 * 1
# Yale Facial Image classes (0-8)
num_classes = 8


class Cnn:
    def __init__(self):
        self.session = []
        self.writer = []
        self.saver = []
        self.merged_summary_op = []
        self.train_step = []
        self.val_summary = []
        self.train_summary = []
        self.loss_val = []
        self.loss = []
        self.logits = []
        self.is_training = []
        self.y_ = []
        self.x_ = []

    def build_graph(self):
        self.x_ = tf.compat.v1.placeholder("float", shape=[None, 50, 50, 1], name='X')
        self.y_ = tf.compat.v1.placeholder("int32", shape=[None, 8], name='Y')
        self.is_training = tf.compat.v1.placeholder(tf.bool)

        with tf.name_scope("model") as scope:
            # layer 1
            conv1 = tf.compat.v1.layers.conv2d(inputs=self.x_, filters=64, kernel_size=[5, 5],
                                               padding="same", activation=None)  # tf.nn.relu
            conv1_bn = tf.compat.v1.layers.batch_normalization(inputs=conv1, axis=-1, momentum=0.9, epsilon=0.001,
                                                               center=True,
                                                               scale=True, training=self.is_training, name='conv1_bn')
            conv1_bn_relu = tf.nn.relu(conv1_bn)
            pool1 = tf.compat.v1.layers.max_pooling2d(inputs=conv1_bn_relu, pool_size=[2, 2], strides=2)

            # layer 2
            conv2 = tf.compat.v1.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5],
                                               padding="same", activation=None)
            conv2_bn = tf.compat.v1.layers.batch_normalization(inputs=conv2, axis=-1, momentum=0.9, epsilon=0.001,
                                                               center=True,
                                                               scale=True, training=self.is_training, name='conv2_bn')
            conv2_bn_relu = tf.nn.relu(conv2_bn)
            pool2 = tf.compat.v1.layers.max_pooling2d(inputs=conv2_bn_relu, pool_size=[2, 2], strides=2)

            # layer 3
            conv3 = tf.compat.v1.layers.conv2d(inputs=pool2, filters=32, kernel_size=[5, 5],
                                               padding="same", activation=None)
            conv3_bn = tf.compat.v1.layers.batch_normalization(inputs=conv3, axis=-1, momentum=0.9, epsilon=0.001,
                                                               center=True,
                                                               scale=True, training=self.is_training, name='conv3_bn')
            conv3_bn_relu = tf.nn.relu(conv3_bn)
            pool3 = tf.compat.v1.layers.max_pooling2d(inputs=conv3_bn_relu, pool_size=[2, 2], strides=2)
            # print(pool3)

            pool3_flat = tf.reshape(pool3, [-1, 4 * 4 * 50])

            # FC layer 1
            FC1 = tf.compat.v1.layers.dense(inputs=pool3_flat, units=128, activation=tf.nn.relu)
            dropout_1 = tf.compat.v1.layers.dropout(inputs=FC1, rate=0.5, training=self.is_training)
            # FC layer 1
            FC2 = tf.compat.v1.layers.dense(inputs=dropout_1, units=64, activation=tf.nn.relu)
            dropout_2 = tf.compat.v1.layers.dropout(inputs=FC2, rate=0.5, training=self.is_training)
            self.logits = tf.compat.v1.layers.dense(inputs=dropout_2, units=8)
            # self.logits = tf.compat.v1.layers.dense(inputs=FC2, units=8)

            with tf.name_scope("loss_func") as scope:
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_))
                self.loss_val = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_))

                # Add loss to tensorboard
                tf.summary.scalar("loss_train", self.loss)
                tf.summary.scalar("loss_val", self.loss_val)
                self.train_summary = tf.compat.v1.summary.scalar("loss_train", self.loss)
                self.val_summary = tf.compat.v1.summary.scalar("loss_val", self.loss_val)

            # Get ops to update moving_mean and moving_variance for batch_norm
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

            with tf.name_scope("optimizer") as scope:
                global_step = tf.Variable(0, trainable=False)
                starter_learning_rate = 1e-3
                # decay every 1000 steps with a base of 0.9
                learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate, global_step,
                                                                     1000, 0.9, staircase=True)

                with tf.control_dependencies(update_ops):
                    self.train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(self.loss,
                                                                                               global_step=global_step)
                tf.summary.scalar("learning_rate", learning_rate)
                tf.summary.scalar("global_step", global_step)

            # Merge op for tensorboard
            self.merged_summary_op = tf.compat.v1.summary.merge_all()

            # Build graph
            init = tf.compat.v1.global_variables_initializer()

            # Saver for checkpoints
            self.saver = tf.compat.v1.train.Saver(max_to_keep=None)

            # it's good to allocate a fraction of memory
            gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
            self.session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

            # summary output configuration logs
            self.writer = tf.compat.v1.summary.FileWriter("./logs/facial3", self.session.graph)
            self.session.run(init)

    def train(self):
        # batch
        batch_size = batch_size_no
        # get data
        pickle_folder_path = os.path.join(in_dir + input_filename)
        X = []
        y = []
        with open(os.path.join(pickle_folder_path, 'image_label'), 'rb') as f:
            img_obj = pickle.load(f)
            # print(img_obj)

            for item in img_obj:
                X.append(item['image'])
                y.append(item['label'])

        # split data
        X_train, X_val, y_train, y_val = train_test_split(X, y)

        # Convert target to binary class vectors
        y_train = tf.keras.utils.to_categorical(y_train, 8)
        y_val = tf.keras.utils.to_categorical(y_val, 8)

        # Handle batches with tf data API
        dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset_train = dataset_train.shuffle(buffer_size=1000)
        dataset_train = dataset_train.repeat()
        dataset_train = dataset_train.batch(batch_size)
        dataset_test = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        dataset_test = dataset_test.repeat()
        dataset_test = dataset_test.batch(batch_size)

        # Create an iterator
        iter_train = dataset_train.make_one_shot_iterator()
        iter_train_op = iter_train.get_next()
        iter_test = dataset_test.make_one_shot_iterator()
        iter_test_op = iter_test.get_next()

        # Build model graph
        self.build_graph()

        # Train Loop
        for i in range(20):
            # get batch with cpu
            with tf.device('/cpu:0'):
                batch_train = self.session.run([iter_train_op])
                batch_x_train, batch_y_train = batch_train[0]
            # get loss
            if i % 100 == 0:
                # get data with cpu
                with tf.device('/cpu:0'):
                    batch_test = self.session.run([iter_test_op])
                    batch_x_test, batch_y_test = batch_test[0]

                loss_train, summary_1 = self.session.run([self.loss, self.merged_summary_op],
                                                         feed_dict={self.x_: batch_x_train,
                                                                    self.y_: batch_y_train,
                                                                    self.is_training: True})

                loss_val, summary_2 = self.session.run([self.loss_val, self.val_summary],
                                                       feed_dict={self.x_: batch_x_test,
                                                                  self.y_: batch_y_test, self.is_training: False})
                print("Loss Train: {0} Loss Val: {1}".format(self.loss, self.loss_val))
                # Write summary to tensorboard
                self.writer.add_summary(summary_1, i)
                self.writer.add_summary(summary_2, i)

            # Execute training ops
            self.train_step.run(session=self.session, feed_dict={
                self.x_: batch_x_train, self.y_: batch_y_train, self.is_training: True})

        # Model saving
        save_dir = os.path.join(out_dir + output_folder)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, "model")
        filename = self.saver.save(self.session, checkpoint_path)
        print("Model saved in file: %s" % filename)


if __name__ == '__main__':
    # import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_eager_execution()
    in_dir = r"/GitHub/face_recognition/output/"  # path to processed signal
    input_filename = 'processed_YaleFaces_5x3r'

    out_dir = "/GitHub/face_recognition/output/"  # path to input target file
    output_folder = 'features_label'
    batch_size_no = 500

    fire.Fire(Cnn)
