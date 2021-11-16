# coding: utf-8
import os, sys

import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow.python.keras import backend as K
from deepctr.utils import SingleFeat

from utils import *
from models import DIN, DIN_All_Fields

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
K.set_session(tf.Session(config=tfconfig))

if __name__ == "__main__":
	FRAC = FRAC
	SESS_MAX_LEN = DIN_SESS_MAX_LEN
	fd = pd.read_pickle(ROOT_DATA+'model_input/din_fd_' +
						str(FRAC) + '_' + str(SESS_MAX_LEN) + '.pkl')
	model_input = pd.read_pickle(
		ROOT_DATA+'model_input/din_input_' + str(FRAC) + '_' + str(SESS_MAX_LEN) + '.pkl')
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
	BATCH_SIZE = 4096

	sess_feature = ['cate_id', 'brand']
	TEST_BATCH_SIZE = 2 ** 14

	print('train len: %d\ttest_len: %d' % (train_label.shape[0], test_label.shape[0]))

	model_type = sys.argv[1]
	for i in range(5):
		if model_type == 'All_Fields':
			print('Start training DIN_All_Fields: ' + str(i))
			log_path = ROOT_DATA + 'log/DIN_All_Fields_log_' + str(i) + '.txt'
			best_model_path = ROOT_DATA + 'best_model/din_all_fields.h5'
			model = DIN_All_Fields(fd, sess_feature, embedding_size=64, att_activation='dice',
						att_weight_normalization=False, hist_len_max=sess_len_max, dnn_hidden_units=(200, 80),
						att_hidden_size=(64, 16,))
		else:
			print('Start training DIN: ' + str(i))
			log_path = ROOT_DATA + 'log/DIN_log_' + str(i) + '.txt'
			best_model_path = ROOT_DATA + 'best_model/din.h5'
			model = DIN(fd, sess_feature, embedding_size=64, att_activation='dice',
						att_weight_normalization=False, hist_len_max=sess_len_max, dnn_hidden_units=(200, 80),
						att_hidden_size=(64, 16,))

		model.compile('adagrad', 'binary_crossentropy')


		hist_ = model.fit(train_input[:], train_label,
						  batch_size=BATCH_SIZE, epochs=10, initial_epoch=0, verbose=0,
						  callbacks=[LossHistory(log_path),
									 auc_callback(training_data=[train_input, train_label],
												  test_data=[test_input, test_label],
												  best_model_path=best_model_path)])

		K.clear_session()
		del model
