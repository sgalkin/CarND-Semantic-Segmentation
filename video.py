#!/usr/bin/env python

import argparse
import moviepy.editor as mpy
import tensorflow as tf
import numpy as np
from train import load_vgg
from train import layers

num_classes = 2

def apply(sess, input, keep_prob, logits, image):
    im_softmax, = sess.run([tf.nn.softmax(logits)],
                           {keep_prob: 1.0, input: [image]})
    image_shape = image.shape
    im_softmax = im_softmax[:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0]]))
    return 0.5*image + 0.5*mask
    
def recode(video, graph, weights):
    with tf.Session() as sess:
        input, keep_prob, vgg_layer3, vgg_layer4, vgg_layer7 = load_vgg(sess, graph)
        output = layers(vgg_layer3, vgg_layer4, vgg_layer7, num_classes)
        logits = tf.reshape(output, (-1, num_classes))
        
        saver = tf.train.Saver()
        saver.restore(sess, wights)
        
        return video.fl_image(lambda i: apply(sess, input, keep_prob, logits, i))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video converseion utility')
    parser.add_argument('video', type=str, help='Input video file')
    parser.add_argument('-o', dest='filename', type=str, required=True, help='Output video file')
    parser.add_argument('-g', dest='graph', type=str, required=True, help='Graph')
    parser.add_argument('-w', dest='weights', type=str, required=True, help='Graph Weights')
    parser.add_argument('-b', type=int, default=0, help='Start time')
    parser.add_argument('-e', type=int, default=-1, help='End time')
    args = parser.parse_args()

    video = mpy.VideoFileClip(args.video)
    video = video.subclip(args.b, args.e)
#    video = video.resize(((576, 160)))
    video = recode(video, args.graph, args.weights)
    video.write_videofile(args.filename, audio=False)
