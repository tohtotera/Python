#-*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model
import input_data_proto2 as input_data

#model parameters.
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 500, 'Batch size')
tf.app.flags.DEFINE_integer('max_steps', 3000, 'Max of training step.')
tf.app.flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
tf.app.flags.DEFINE_string('dataset_dir', None, 'Directory of images.')

def run_training():
	"""ニューラルネットワークの学習を行う
	"""
	#教師データの読み込み
	data_sets = input_data.read_data_sets(FLAGS.dataset_dir)

	with tf.Graph().as_default():
		with tf.Session() as sess:
			images_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
			labels_placeholder = tf.placeholder(tf.float32, shape=[None, 10])
			keep_prob = tf.placeholder(tf.float32)

			logits = model.inference(images_placeholder, keep_prob)
			loss_value = model.loss(logits, labels_placeholder)
			train_op = model.training(loss_value)
			accuracy = model.evaluation(logits, labels_placeholder)

			sess.run(tf.initialize_all_variables()) #r0.12ではtf.global_variables_initializer()を使う
			saver = tf.train.Saver()

			#学習過程を可視化するためSummaryWriterを使う
			train_writer = tf.train.SummaryWriter(FLAGS.train_dir + '/train', sess.graph) 
			test_writer = tf.train.SummaryWriter(FLAGS.train_dir + '/test') 
			loss_summary = tf.scalar_summary('loss', loss_value) 
			acc_summary = tf.scalar_summary('accuracy', accuracy) 

			#訓練
			for step in xrange(FLAGS.max_steps):
				train_images, train_labels = data_sets.train.next_batch(FLAGS.batch_size)
				feed_dict = {
					images_placeholder: train_images,
					labels_placeholder: train_labels,
					keep_prob: 0.5}
				_, cross_entropy = sess.run([train_op, loss_value], feed_dict=feed_dict)

				loss_str, acc_str = sess.run([loss_summary, acc_summary], feed_dict=feed_dict)
				train_writer.add_summary(loss_str, step)
				train_writer.add_summary(acc_str, step)

				if step%100 == 0 or (step + 1) == FLAGS.max_steps:
					#テストデータを使って評価
					feed_dict = {
						images_placeholder: data_sets.test.images,
						labels_placeholder: data_sets.test.labels,
						keep_prob: 1.0}
					test_acc, acc_str = sess.run([accuracy, acc_summary], feed_dict=feed_dict)
					test_writer.add_summary(acc_str, step)

					print('step %d, cross_entropy %g, TEST_ACCURACY %g'%(step, cross_entropy, test_acc))

					#モデルを保存
					saver.save(sess, 'model.ckpt', global_step=step)

def main(_):
	if not FLAGS.dataset_dir:
		raise ValueError('You must supply the dataset directory with --dataset_dir')
	else:
		run_training()

if __name__ == '__main__':
	tf.app.run()
