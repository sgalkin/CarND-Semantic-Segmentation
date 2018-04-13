#!/usr/bin/env python

import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

L2_WEIGHT = 1e-3
STDDEV = 1e-2

LEARNING_RATE=0.0005
KEEP_PROB=0.5

EPOCHS = 30
BATCH_SIZE = 34


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), \
       'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    return tuple(graph.get_tensor_by_name(x)
                 for x in (vgg_input_tensor_name,
                           vgg_keep_prob_tensor_name,
                           vgg_layer3_out_tensor_name,
                           vgg_layer4_out_tensor_name,
                           vgg_layer7_out_tensor_name))
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    output = vgg_layer7_out
    output = tf.stop_gradient(output)
    
    output = tf.layers.conv2d(output, num_classes,
                              kernel_size=(1, 1), strides=(1, 1), padding='SAME',
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_WEIGHT),
                              kernel_initializer=tf.truncated_normal_initializer(stddev=STDDEV))
    output = tf.layers.conv2d_transpose(output, num_classes,
                                        kernel_size=(4, 4), strides=(2, 2), padding='SAME',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_WEIGHT),
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=STDDEV))
    output = tf.add(output,
                    tf.layers.conv2d(vgg_layer4_out, num_classes,
                                     kernel_size=(1, 1), strides=(1, 1), padding='SAME',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_WEIGHT),
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=STDDEV)))
    output = tf.layers.conv2d_transpose(output, num_classes,
                                        kernel_size=(4, 4), strides=(2, 2), padding='SAME',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_WEIGHT),
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=STDDEV))
    output = tf.add(output,
                    tf.layers.conv2d(vgg_layer3_out, num_classes,
                                     kernel_size=(1, 1), strides=(1, 1), padding='SAME',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_WEIGHT),
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=STDDEV)))
    output = tf.layers.conv2d_transpose(output, num_classes,
                                        kernel_size=(16, 16), strides=(8, 8), padding='SAME',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_WEIGHT),
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=STDDEV))
    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = correct_label #tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    #TODO: tune
    regularizer = 1
    regularizer_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    train = optimizer.minimize(cross_entropy_loss + regularizer*regularizer_loss)

    prediction = tf.argmax(nn_last_layer, axis=3)
    truth = correct_label[:, :, :, 1]
    iou, confusion = tf.metrics.mean_iou(truth, prediction, num_classes)
                                                    
    return logits, train, cross_entropy_loss, iou, confusion
tests.test_optimize(optimize)


def train_nn(sess, saver,
             epochs, batch_size, get_batches_fn,
             train_op, cross_entropy_loss,
             input_image, correct_label,
             keep_prob,
             learning_rate,
             iou,
             confusion):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param saver: TF Saver
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
        
    print("Training...")
    for i in range(epochs):
        print("Epoch: {} / {} ...".format(i + 1, epochs))
        for image, label in get_batches_fn(batch_size):
            logits, loss = sess.run([train_op, cross_entropy_loss],
                                    feed_dict={input_image: image,
                                               correct_label: label,
                                               #TODO: tune
                                               keep_prob: KEEP_PROB,
                                               learning_rate: LEARNING_RATE})
            
            sess.run(confusion,
                     feed_dict={input_image: image,
                                correct_label: label,
                                keep_prob: 1.0})
            mean_iou = sess.run(iou,
                                feed_dict={input_image: image,
                                           correct_label: label,
                                           keep_prob: 1.0})
            print("Loss: {:.6f}; IoU: {:6f}".format(loss, mean_iou)) #tf.metrics.mean_iou(label, logits, 2)))
        print()
        if saver is not None:
            pass
            #saver.save(sess, 'model', global_step=i)

tests.test_train_nn(train_nn)

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    augment_dir = '/tmp/data_road'
    runs_dir = '/tmp/runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    helper.augment(os.path.join(data_dir, 'data_road/training'), augment_dir, image_shape)

    with tf.Session() as sess:
 
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(augment_dir, None) #os.path.join(data_dir, 'data_road/training'), image_shape)
                                                                                                                
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        input, keep_prob, vgg_layer3, vgg_layer4, vgg_layer7 = load_vgg(sess, vgg_path)
        output = layers(vgg_layer3, vgg_layer4, vgg_layer7, num_classes)
        saver = tf.train.Saver()

        # Train NN using the train_nn function
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)

        logits, train_op, cross_entropy_loss, iou, confusion = optimize(output, correct_label, learning_rate, num_classes)

        train_nn(sess, saver, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss,
                 input, correct_label, keep_prob, learning_rate, iou, confusion)
    
        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input)
        
        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
