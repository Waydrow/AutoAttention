
import tensorflow as tf
import time
from sklearn.metrics import log_loss, roc_auc_score
from collections import defaultdict
import numpy as np

FRAC = 0.25

DIN_SESS_MAX_LEN = 50

DSIN_SESS_COUNT = 5
DSIN_SESS_MAX_LEN = 10

ROOT_DATA = '../data/'

def cal_group_auc(labels, preds, user_id_list):
	"""Calculate group auc"""

	if len(user_id_list) != len(labels):
		raise ValueError(
			"impression id num should equal to the sample num," \
			"impression id num is {0}".format(len(user_id_list)))
	group_score = defaultdict(lambda: [])
	group_truth = defaultdict(lambda: [])
	for idx, truth in enumerate(labels):
		user_id = user_id_list[idx]
		score = preds[idx]
		truth = labels[idx]
		group_score[user_id].append(score)
		group_truth[user_id].append(truth)

	group_flag = defaultdict(lambda: False)
	for user_id in set(user_id_list):
		truths = group_truth[user_id]
		flag = False
		for i in range(len(truths) - 1):
			if truths[i] != truths[i + 1]:
				flag = True
				break
		group_flag[user_id] = flag

	impression_total = 0
	total_auc = 0
	#
	for user_id in group_flag:
		if group_flag[user_id]:
			auc = roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))
			total_auc += auc * len(group_truth[user_id])
			impression_total += len(group_truth[user_id])
	group_auc = float(total_auc) / impression_total
	group_auc = round(group_auc, 4)
	return group_auc

class LossHistory(tf.keras.callbacks.Callback):
	def __init__(self, path):
		self.path = path
		super(LossHistory).__init__()

	def on_train_begin(self,logs=None):
		# self.start_time = time.time()
		self.losses = []

	def on_batch_end(self,batch,logs=None):
		self.losses.append(round(logs.get('loss'), 4))

	def on_train_end(self, logs=None):
		# print('train_end, cost time: %.2f' % (time.time()-self.start_time))
		with open(self.path, 'w') as f:
			for loss in self.losses:
				f.write(str(loss)+'\n')


class auc_callback(tf.keras.callbacks.Callback):
	def __init__(self, training_data, test_data, best_model_path, is_prun=False, target_sparse=0.5):
		self.x = training_data[0]
		self.y = training_data[1]
		self.x_test = test_data[0]
		self.y_test = test_data[1]
		self.best_model_path = best_model_path

		self.is_prun = is_prun
		self.iter = 0
		self.target_sparse = target_sparse
		self.weights = None
		self.prun_layer = 'auto_attention__layer_1'
		super(auc_callback, self).__init__()


	def on_train_begin(self, logs={}):
		self.start_time = time.time()
		self.best_auc = 0.
		self.test_loss = 0.

		if self.is_prun:
			print('Target_sparse rate: ' + str(self.target_sparse))
		return

	def on_train_end(self, logs={}):
		('Test loss: %.4f\tBest Test AUC: %.4f\tCost time: %d' % (self.test_loss, self.best_auc, time.time()-self.start_time))
		# if self.is_prun:
		# 	print(self.model.get_layer(self.prun_layer).get_weights())
		return

	def on_epoch_begin(self, epoch, logs={}):
		self.epoch_start_time = time.time()
		return

	def on_epoch_end(self, epoch, logs={}):
		# y_pred = self.model.predict(self.x, 4096)
		# train_loss = log_loss(self.y, y_pred)
		# train_auc = roc_auc_score(self.y, y_pred)
		train_loss = logs.get('loss')

		# if self.is_prun:
		# 	self.model.get_layer(self.prun_layer).set_weights(self.weights)
		y_pred_test = self.model.predict(self.x_test, 2 ** 14)
		test_loss = log_loss(self.y_test, y_pred_test)
		# test_auc = roc_auc_score(self.y_test, y_pred_test)
		test_auc = cal_group_auc(self.y_test, np.squeeze(y_pred_test), np.squeeze(self.x_test[1]))
		print('Epoch: %d\ttrain loss: %.4f\ttest loss: %.4f\ttest auc: %.4f\tCost time: %d' %
			  (epoch, train_loss, test_loss, test_auc, time.time() - self.epoch_start_time))

		if self.best_auc < test_auc:
			self.best_auc = test_auc
			self.test_loss = test_loss
			# self.model.save_weights(self.best_model_path)
		else:
			self.model.stop_training = True
		return

	def on_batch_begin(self, batch, logs={}):
		self.iter += 1
		return

	def on_batch_end(self, batch, logs={}):
		if self.is_prun and self.iter % 100 == 0:
			self.adaptive_sparse = self.target_sparse * (1 - 0.8**(self.iter /100.))
			layer = self.model.get_layer(self.prun_layer)
			weights = layer.get_weights()
			# print('********** Iter: ' + str(self.iter))
			# print(weights)
			fields_num = weights[0].shape[0]
			prun_num = round(self.adaptive_sparse * fields_num)
			tmp_weights = []
			for i in weights[0]:
				tmp_weights.append(abs(i[0]))

			threshold = sorted(tmp_weights)[:prun_num][-1]
			for i, v in enumerate(weights[0]):
				if abs(v[0]) <= threshold:
					weights[0][i][0] = 0

			layer.set_weights(weights)
			self.weights = weights

			# if self.iter == 1300:
			# 	print(self.model.get_layer(self.prun_layer).get_weights())
			# print(layer.get_weights)
		return


class prun_callback(tf.keras.callbacks.Callback):
	def __init__(self):
		self.iter = 0
		self.target_sparse = 0.5
		self.weights = None
		# self.fields_num =
		self.prun_layer = 'fw_fm__layer_new_1'
		self.stop_prun = False
		super(prun_callback, self).__init__()

	def on_batch_begin(self, batch, logs=None):
		self.iter += 1
		return

	def on_batch_end(self, batch, logs=None):
		if not (self.stop_prun) and self.iter % 100 == 0:
			self.adaptive_sparse = self.target_sparse * (1 - 0.8**(self.iter /100.))
			layer = self.model.get_layer(self.prun_layer)
			weights = layer.get_weights()
			# print('********** Iter: ' + str(self.iter))
			# print(weights)
			fields_num = weights[0].shape[0]
			prun_num = round(self.adaptive_sparse * fields_num)
			tmp_weights = []
			for i in weights[0]:
				tmp_weights.append(abs(i[0]))

			threshold = sorted(tmp_weights)[:prun_num][-1]
			for i, v in enumerate(weights[0]):
				if abs(v[0]) <= threshold:
					weights[0][i][0] = 0

			layer.set_weights(weights)
			if self.iter % 1300 == 0:
				layer.trainable = False
				self.stop_prun = True
				self.model.compile('adagrad', 'binary_crossentropy')
				self.weights = weights
			# print(layer.get_weights)
		return

	def on_epoch_end(self, epoch, logs=None):
		# print(self.model.get_layer('fw_fm__layer_new_1').trainable)
		return

	def on_train_end(self, logs=None):
		layer = self.model.get_layer(self.prun_layer)
		print(layer.get_weights())
		print('Trainable')
		print(self.model.get_layer(self.prun_layer).trainable)
		return