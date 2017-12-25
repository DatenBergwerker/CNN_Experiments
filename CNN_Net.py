import logging
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from sklearn.model_selection import train_test_split 


def unpickle(file):
    """Unpack CIFAR 10 Images to array."""
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def cnn_model_fn(features, labels, mode, params):
    """Model function for CNN."""

    # Input Layer
    input_layer = tf.reshape(features, [-1, IMG_SIZE, IMG_SIZE, COL_CHANNELS])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=32,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu,
                             strides=2)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu,
                             strides=2)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    height = int(pool2.get_shape()[1])
    width = int(pool2.get_shape()[2])
    pool2_flat = tf.reshape(pool2, [-1, width * height * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=params["dropout"], training=mode == learn.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    loss = None
    train_op = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=params["learnrate"],
            optimizer=params["optimizer"])

    # Generate Predictions
    predictions = {"classes": tf.argmax(input=logits, axis=1),
                   "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)


if __name__ == "__main__":
    # Flags and parameters
    PATH = ""
    IMG_SIZE = 32
    COL_CHANNELS = 3
    FEATURES = COL_CHANNELS * IMG_SIZE ** 2
    N_SAMPLE = 50000
    RESTORE = False

    params = {"optimizer": "SGD",
              "learnrate": 0.001,
              "dropout": 0.4}

    logging.basicConfig(filename="CNN.log",
                        level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%d.%m.%Y %H:%M:%S")

    logging.info("Script Start: CIFAR 10 CNN, PARAMS: ")

    # Data import loop
    data = np.zeros((N_SAMPLE, FEATURES + 1))
    for i in range(0, 5):
        unpick = unpickle(f"{PATH}/cifar-10-batches-py/data_batch_{i + 1}")
        mat = unpick[b"data"]
        labels = unpick[b"labels"]
        data[i * 10000:(i + 1) * 10000, 0:data.shape[1] - 1] = mat
        data[i * 10000:(i + 1) * 10000, -1] = labels

    test_data = np.zeros(shape=(10000, FEATURES + 1))
    unpick = unpickle(f"{PATH}/cifar-10-batches-py/test_batch")
    X_test = unpick[b"data"][:, : -1]
    X_test = X_test.astype("float32")
    Y_test = unpick[b"labels"][:, -1]
    Y_test = Y_test.astype("float32")

    X_train = data[:, :-1]
    X_train = X_train.astype("float32")
    Y_train = data[:, -1]
    Y_train = Y_train.astype("float32")

    # Create the Estimator
    test_classifier = learn.Estimator(model_fn=cnn_model_fn,
                                      params=params,
                                      model_dir=f"{PATH}/CNN_test")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                              every_n_iter=50)

    # Configure the accuracy metric for evaluation
    metrics = {"accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes"), }

    # Model training
    if not RESTORE:
        logging.info("Training Start")
        test_classifier.fit(x=X_train,
                            y=Y_train,
                            batch_size=100,
                            steps=20000,
                            monitors=[logging_hook])
        logging.info("Training End")

    # Model evaluation
    logging.info("Test Start")
    test_classifier.evaluate(x=X_test,
                             y=Y_test)
    logging.info("Test End")
