# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def layer_norm(inputs, num_splits=1, bias_start=0.0, scope="layer_norm"):
  with tf.variable_scope(scope):
    '''for clarification of shapes:
    inputs = [batch_size, num_units]
    mean = [batch_size]
    variance = [batch_size]
    alpha = [num_units]
    bias = [num_units]
    output = [batch_size, num_units]
    '''
    num_units = inputs.get_shape().as_list()[1] // num_splits

    alpha = tf.get_variable('alpha', [num_splits, num_units],
                            initializer=tf.constant_initializer(1.0))
    bias = tf.get_variable('bias', [num_splits, num_units],
                           initializer=tf.constant_initializer(bias_start))

    new_inputs = tf.reshape(inputs, [-1, num_splits, num_units])
    mean, variance = moments_for_layer_norm(new_inputs, axes=[2])
    outputs = (alpha * (new_inputs - mean)) / variance + bias

    return tf.reshape(outputs, [-1, num_splits * num_units])


def moments_for_layer_norm(x, axes, name=None, epsilon=0.001):
  '''output for mean and variance should be [batch_size]'''

  if not isinstance(axes, list): axes = list(axes)

  with tf.name_scope(values=[x, axes], name=name, default_name="moments"):
    mean = tf.reduce_mean(x, axes, keep_dims=True)
    variance = tf.sqrt(tf.reduce_mean(tf.square(x - mean), axes, keep_dims=True) + epsilon)

  return mean, variance
