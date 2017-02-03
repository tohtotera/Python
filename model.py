#-*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def inference(images, keep_prob):
	#重みを標準偏差0.1の正規分布で初期化
	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial, name='weights')

	#バイアスを0.1で初期化
	def bias_variable(shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial, name='biases')

	#畳み込み層の作成
	def conv2d(x, W):
		return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

	#プーリング層の作成
	def max_pool_2x2(x):
		return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	#畳み込み層1
	with tf.name_scope('conv1') as scope:
		W_conv1 = weight_variable([5, 5, 3, 32])
		b_conv1 = bias_variable([32])
		h_conv1 = tf.nn.relu(conv2d(images, W_conv1) + b_conv1)

	#プーリング層1
	with tf.name_scope('pool1') as scope:
		h_pool1 = max_pool_2x2(h_conv1)

	#畳み込み層2
	with tf.name_scope('conv2') as scope:
		W_conv2 = weight_variable([5, 5, 32, 64])
		b_conv2 = bias_variable([64])
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

	#プーリング層2
	with tf.name_scope('pool2') as scope:
		h_pool2 = max_pool_2x2(h_conv2)

	#全結合層1
	with tf.name_scope('fc1') as scope:
		W_fc1 = weight_variable([8 * 8 * 64, 1024])
		b_fc1 = bias_variable([1024])
		h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

		#dropoutの設定
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	#全結合層2
	with tf.name_scope('fc2') as scope:
		W_fc2 = weight_variable([1024, 10])
		b_fc2 = bias_variable([10])

		logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	return logits

def loss(logits, labels):
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
	loss = tf.reduce_mean(cross_entropy)
	return loss

def training(total_loss):
	train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss)
	return train_step

def evaluation(logits, labels):
	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return accuracy

