#-*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cPickle
import os
import collections
import cv2

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32

Datasets = collections.namedtuple('Datasets', ['train', 'test'])

class Dataset(object):
	def __init__(self, images, labels):
		self._num_examples = len(images)
		self._images = images
		self._labels = labels
		self._epochs_completed = 0
		self._index_in_epoch = 0

	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size):
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			self._epochs_completed += 1
			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self._images = self._images[perm]
			self._labels = self._labels[perm]

			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		end = self._index_in_epoch
		return self._images[start:end], self._labels[start:end]


def dense_to_one_hot(labels_dense, num_classes):
	num_labels = len(labels_dense)
	labels_one_hot = np.zeros((num_labels, num_classes))
	for i in xrange(num_labels):
		labels_one_hot[i][labels_dense[i]] = 1
	return labels_one_hot
	
def _convert_rgb_image(rgb_data, height, width):
	return np.c_[rgb_data[0], rgb_data[1], rgb_data[2]].reshape((height, width, 3))

def convert_images(images, height, width):
	cvt_images = []
	for i in xrange(0,len(images),9):
		top = _convert_rgb_image(images[i:i+3], height, width)
		bottom = _convert_rgb_image(images[i+3:i+6], height, width)
		side = _convert_rgb_image(images[i+6:i+9], height, width)

		#bottom画像を上下反転
		bottom = cv2.flip(bottom, 0)

		#トリミング,サイズ変更
		#入力解像度が32x32なのでやらない

		#重ね合わせ
		top = cv2.cvtColor(top, cv2.COLOR_RGB2GRAY)
		bottom = cv2.cvtColor(bottom, cv2.COLOR_RGB2GRAY)
		side = cv2.cvtColor(side, cv2.COLOR_RGB2GRAY)
		merge = np.c_[top.flatten(), bottom.flatten(), side.flatten()].reshape((height, width,3))

		cvt_images.append(merge)
	return cvt_images

def read_data_sets(data_dir):
	meta_file = os.path.join(data_dir, 'batches.meta')
	with open(meta_file, 'rb') as f:
		meta = cPickle.load(f)

	batch_file = os.path.join(data_dir, 'data_batch_%d')
	#batch_1〜5は学習用データ
	train_images = []
	train_labels = []
	for i in xrange(1,5):
		with open(batch_file%(i), 'rb') as f:
			data = cPickle.load(f)
			train_images.extend(convert_images(data['data'], IMAGE_HEIGHT, IMAGE_WIDTH))
			train_labels.extend(dense_to_one_hot(data['labels'], len(meta['label_names'])))

	#batch_6はテスト用データ
	with open(batch_file%(6), 'rb') as f:
		data = cPickle.load(f)
		test_images = convert_images(data['data'], IMAGE_HEIGHT, IMAGE_WIDTH)
		test_labels = dense_to_one_hot(data['labels'], len(meta['label_names']))
	
	#RGB値は[0.0-1.0]に正規化
	train_images = np.array(train_images, dtype=np.float32) / 255.0
	test_images = np.array(test_images, dtype=np.float32) / 255.0

	train_labels = np.array(train_labels)
	test_labels = np.array(test_labels)

	train = Dataset(train_images, train_labels)
	test = Dataset(test_images, test_labels)
	return Datasets(train=train, test=test)

