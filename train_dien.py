# coding: utf-8
import os

import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow.python.keras import backend as K
from deepctr.utils import SingleFeat

from utils import *
from models import DIEN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
K.set_session(tf.Session(config=tfconfig))

if __name__ == "__main__":

	FRAC = FRAC
	SESS_MAX_LEN = DIN_SESS_MAX_LEN
	fd = pd.read_pickle(ROOT_DATA+'model_input/dien_fd_' +
						str(FRAC) + '_' + str(SESS_MAX_LEN) + '.pkl')
	model_input = pd.read_pickle(
		ROOT_DATA+'model_input/dien_input_' + str(FRAC) + '_' + str(SESS_MAX_LEN) + '.pkl')
	label = pd.read_pickle(ROOT_DATA+'model_input/dien_label_' +
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

	DIEN_NEG_SAMPLING = True
	for i in range(5):
		model = DIEN(fd, sess_feature, 64, sess_len_max, "AUGRU", att_hidden_units=(64, 16),
					 att_activation='sigmoid', use_negsampling=DIEN_NEG_SAMPLING, att_type='base')

		model.compile('adagrad', 'binary_crossentropy')

		if DIEN_NEG_SAMPLING:
			print('Start training DIEN_neg: ' + str(i))
			log_path = ROOT_DATA + 'log/DIEN_neg_log_' + str(i) + '.txt'
			best_model_path = ROOT_DATA + 'best_model/dien_neg.h5'
			hist_ = model.fit(train_input[:], train_label,
							  batch_size=BATCH_SIZE, epochs=10, initial_epoch=0, verbose=0,
							  callbacks=[LossHistory(log_path),
										 auc_callback(training_data=[train_input, train_label],
													  test_data=[test_input, test_label],
													  best_model_path=best_model_path)])
		else:
			print('Start training DIEN: ' + str(i))
			log_path = ROOT_DATA + 'log/DIEN_log_' + str(i) + '.txt'
			best_model_path = ROOT_DATA + 'best_model/dien.h5'
			hist_ = model.fit(train_input[:-3] + train_input[-1:], train_label, batch_size=BATCH_SIZE, epochs=10,
							  initial_epoch=0, verbose=2, callbacks=[LossHistory(log_path),
																	 auc_callback(training_data=[train_input, train_label],
																				  test_data=[test_input, test_label],
																				  best_model_path=best_model_path)])

		K.clear_session()
		del model
