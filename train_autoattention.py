# coding: utf-8
import os, sys

import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow.python.keras import backend as K
from deepctr.utils import SingleFeat


from utils import *
from models import AutoAttention, DotProduct

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
K.set_session(tf.Session(config=tfconfig))

def binary_crossentropy(y_true, y_pred):
	return K.mean(K.binary_crossentropy(y_true, y_pred + 1e-5), axis=-1)

if __name__ == "__main__":
	FRAC = FRAC
	SESS_MAX_LEN = DIN_SESS_MAX_LEN
	fd = pd.read_pickle(ROOT_DATA+'model_input/din_fd_' +
						str(FRAC) + '_' + str(SESS_MAX_LEN) + '.pkl')
	key_length = pd.read_pickle(
		ROOT_DATA+'model_input/din_input_len_' + str(FRAC) + '_' + str(SESS_MAX_LEN) + '.pkl')
	model_input = pd.read_pickle(
		ROOT_DATA+'model_input/din_input_' + str(FRAC) + '_' + str(SESS_MAX_LEN) + '.pkl')
	model_input += key_length
	label = pd.read_pickle(ROOT_DATA+'model_input/din_label_' +
						   str(FRAC) + '_' + str(SESS_MAX_LEN) + '.pkl')

	sample_sub = pd.read_pickle(
		ROOT_DATA+'sampled_data/raw_sample_' + str(FRAC) + '.pkl')

	sample_sub['idx'] = list(range(sample_sub.shape[0]))
	train_idx = sample_sub.loc[sample_sub.time_stamp <
							   1494633600, 'idx'].values
	test_idx = sample_sub.loc[sample_sub.time_stamp >=
							  1494633600, 'idx'].values

	train_input = [i[train_idx] for i in model_input]
	test_input = [i[test_idx] for i in model_input]
	train_label = label[train_idx]
	test_label = label[test_idx]

	sess_len_max = SESS_MAX_LEN
	sess_feature = ['cate_id', 'brand']

	BATCH_SIZE = 4096
	TEST_BATCH_SIZE = 2 ** 14

	print('train len: %d\ttest_len: %d' % (train_label.shape[0], test_label.shape[0]))

	is_prun = False
	sparse_rate = 0.5
	model_type = sys.argv[1]
	for i in range(5):
		print('########################################')
		if model_type == 'DotProduct':
			print('Start training DotProduct: ' + str(i))
			log_path = ROOT_DATA + 'log/DotProduct_log_' + str(i) + '.txt'
			best_model_path = ROOT_DATA + 'best_model/DotProduct.h5'
			model = DotProduct(fd, sess_feature, embedding_size=16, hist_len_max=sess_len_max)
		elif model_type == 'AutoAttention':
			print('Start training AutoAttention: ' + str(i))
			log_path = ROOT_DATA + 'log/AutoAttention_log_' + str(i) + '.txt'
			best_model_path = ROOT_DATA + 'best_model/AutoAttention.h5'
			model = AutoAttention(fd, sess_feature, embedding_size=16, hist_len_max=sess_len_max)
		elif model_type == 'AutoAttention_Prun':
			is_prun = True
			sparse_rate = float(sys.argv[2])
			print('Start training AutoAttention_Prun: ' + str(i))
			log_path = ROOT_DATA + 'log/AutoAttention_Prun_log_' + str(i) + '.txt'
			best_model_path = ROOT_DATA + 'best_model/AutoAttention_Prun.h5'
			model = AutoAttention(fd, sess_feature, embedding_size=16, hist_len_max=sess_len_max)
		else:
			print("Wrong argument model type!")
			sys.exit(0)

		# model.compile(optimizer='adagrad', loss='binary_crossentropy')
		opt = tf.keras.optimizers.Adagrad(lr=0.01)
		model.compile(optimizer=opt, loss=binary_crossentropy)

		hist_ = model.fit(train_input[:], train_label,
						  batch_size=BATCH_SIZE, epochs=10, initial_epoch=0, verbose=0,
						  callbacks=[LossHistory(log_path),
									 auc_callback(training_data=[train_input, train_label],
												  test_data=[test_input, test_label],
												  best_model_path=best_model_path,
												  is_prun=is_prun, target_sparse=sparse_rate)])

		K.clear_session()
		del model
