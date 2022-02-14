# -*- coding:utf-8 -*-

"""
Author: Zuowu Zheng, wozhengzw@gmail.com
Date: March 11, 2021
"""
from tensorflow.python.keras import backend as K
from collections import OrderedDict
import tensorflow as tf
from deepctr.input_embedding import get_inputs_list, create_singlefeat_inputdict, get_embedding_vec_list
from deepctr.layers.core import DNN, PredictionLayer
from deepctr.layers.utils import concat_fun, NoMask
from deepctr.utils import check_feature_config_dict
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import Input, Dense, Embedding, Concatenate, Flatten, Layer, Add, Lambda, PReLU, Average
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from deepctr.layers.sequence import SequencePoolingLayer, positional_encoding
from deepctr.layers import LayerNormalization
import time


def get_input(feature_dim_dict, seq_feature_list, seq_max_len):
	sparse_input, dense_input = create_singlefeat_inputdict(feature_dim_dict)
	user_behavior_input = OrderedDict()
	for i, feat in enumerate(seq_feature_list):
		user_behavior_input[feat] = Input(shape=(seq_max_len,), name='seq_' + str(i) + '-' + feat)
	user_behavior_length = Input(shape=(1,), name='seq_length')

	return sparse_input, dense_input, user_behavior_input, user_behavior_length


class DotProduct_Layer(Layer):
	def __init__(self, maxlen, emb_dim, embed_reg=1e-4, weight_softmax=True, is_keys_diff=False, return_score=False):
		super(DotProduct_Layer, self).__init__()
		self.bias = self.add_weight(shape=(1,), initializer=tf.keras.initializers.RandomNormal(),
									regularizer=tf.keras.regularizers.l2(embed_reg),
									dtype=tf.float32, name='b_fm', trainable=True)

		self.W_query = self.add_weight(name='query', shape=(emb_dim, emb_dim),
									 dtype=tf.float32,
									 initializer=tf.keras.initializers.RandomNormal(),
									 regularizer=tf.keras.regularizers.l2(embed_reg),
									 trainable=True)
		self.W_key = self.add_weight(name='key', shape=(emb_dim, emb_dim),
									 dtype=tf.float32,
									 initializer=tf.keras.initializers.RandomNormal(),
									 regularizer=tf.keras.regularizers.l2(embed_reg),
									 trainable=True)
		self.W_value = self.add_weight(name='value', shape=(emb_dim, emb_dim),
									   dtype=tf.float32,
									   initializer=tf.keras.initializers.RandomNormal(),
									   regularizer=tf.keras.regularizers.l2(embed_reg),
									   trainable=True)

		self.maxlen = maxlen
		self.emb_dim = emb_dim
		self.weight_softmax = weight_softmax
		self.is_keys_diff = is_keys_diff
		self.return_score = return_score

	def build(self, input_shape):
		super(DotProduct_Layer, self).build(input_shape)

	def call(self, inputs, **kwargs):
		# q (None, m, K), all field, m is the number of fields
		# k/v (None, maxlen, K), hist
		# keys_length (None, 1)
		q, k, v, keys_length = inputs
		if self.is_keys_diff:
			q = tf.tensordot(q, self.W_query, axes=1)
			v = tf.tensordot(k, self.W_value, axes=1)
			k = tf.tensordot(k, self.W_key, axes=1)

		q = tf.expand_dims(q, axis=1)  # (None, 1, m, K)
		q = tf.tile(q, multiples=[1, k.shape[1], 1, 1])  # (None, maxlen, m, K)

		k = tf.expand_dims(k, axis=2)  # (None, maxlen, 1, K)
		k = tf.tile(k, multiples=[1, 1, q.shape[2], 1])  # (None, maxlen, m, K)
		a = q * k
		a = tf.reduce_sum(a, axis=2) + self.bias  # (None, maxlen, K)
		a = tf.reduce_sum(a, axis=-1)  # (None, maxlen)
		a = tf.nn.sigmoid(a)  # (None, maxlen)

		# mask
		key_masks = tf.sequence_mask(keys_length[:, 0], self.maxlen)  # (None, maxlen)
		if self.weight_softmax:
			paddings = tf.ones_like(a) * (-2 ** 32 + 1)
		else:
			paddings = tf.zeros_like(a)

		a = tf.where(key_masks, a, paddings)

		a = a / (self.emb_dim ** 0.5)

		# softmax
		if self.weight_softmax:
			a = tf.nn.softmax(logits=a)
		outputs = tf.expand_dims(a, axis=1)  # (None, 1, maxlen)

		if not self.return_score:
			outputs = tf.matmul(outputs, v)  # (None, 1, K)
		# outputs = tf.squeeze(outputs, axis=1)  # (None, K)

		return outputs

	def compute_output_shape(self, input_shape):
		return (None, 1, input_shape[0][-1])

	def compute_mask(self, inputs, mask=None):
		return None

class AutoAttention_Layer(Layer):
	def __init__(self, maxlen, emb_dim, embed_reg=1e-4, weight_softmax=True, is_keys_diff=False, return_score=False):
		super(AutoAttention_Layer, self).__init__()
		self.maxlen = maxlen
		self.emb_dim = emb_dim
		self.embed_reg = embed_reg
		self.weight_softmax = weight_softmax
		self.is_keys_diff = is_keys_diff
		self.return_score = return_score

	def build(self, input_shape):
		q_shape = input_shape[0].as_list()
		self.field_strengths = self.add_weight(name='field_pair_strengths',
											   shape=(q_shape[1], 1),
											   initializer=tf.keras.initializers.RandomNormal(),
											   regularizer=tf.keras.regularizers.l2(self.embed_reg),
											   dtype=tf.float32,
											   trainable=True)

		self.bias = self.add_weight(shape=(1,), initializer=tf.keras.initializers.RandomNormal(),
									dtype=tf.float32, name='b_fwfm',
									regularizer=tf.keras.regularizers.l2(self.embed_reg),
									trainable=True)

		self.W_query = self.add_weight(name='query', shape=(self.emb_dim, self.emb_dim),
									   dtype=tf.float32,
									   initializer=tf.keras.initializers.RandomNormal(),
									   regularizer=tf.keras.regularizers.l2(self.embed_reg),
									   trainable=True)
		self.W_key = self.add_weight(name='key', shape=(self.emb_dim, self.emb_dim),
									 dtype=tf.float32,
									 initializer=tf.keras.initializers.RandomNormal(),
									 regularizer=tf.keras.regularizers.l2(self.embed_reg),
									 trainable=True)
		self.W_value = self.add_weight(name='value', shape=(self.emb_dim, self.emb_dim),
									   dtype=tf.float32,
									   initializer=tf.keras.initializers.RandomNormal(),
									   regularizer=tf.keras.regularizers.l2(self.embed_reg),
									   trainable=True)
		super(AutoAttention_Layer, self).build(input_shape)

	def call(self, inputs, **kwargs):
		# q (None, m, K), all field, m is the number of fields
		# k/v (None, maxlen, K), hist
		# keys_length (None, 1)

		q, k, v, kes_length = inputs
		# emb_dim = q.shape[-1]

		if self.is_keys_diff:
			q = tf.tensordot(q, self.W_query, axes=1)
			v = tf.tensordot(k, self.W_value, axes=1)
			k = tf.tensordot(k, self.W_key, axes=1)

		q = tf.expand_dims(q, axis=1) # (None, 1, m, K)

		q = tf.tile(q, multiples=[1,k.shape[1],1,1]) # (None, maxlen, m, K)

		k = tf.expand_dims(k, axis=2) # (None, maxlen, 1, K)
		k = tf.tile(k, multiples=[1,1,q.shape[2],1]) # (None, maxlen, m, K)
		a = q*k
		s = tf.expand_dims(self.field_strengths, axis=0) # (1, m, 1)
		s = tf.tile(s, multiples=[self.maxlen, 1 ,a.shape[-1]]) # (maxlen, m, K)
		a = a*s
		a = tf.reduce_sum(a, axis=2) + self.bias # (None, maxlen, K)
		a = tf.reduce_sum(a, axis=-1) # (None, maxlen)
		a = tf.nn.sigmoid(a) # (None, maxlen)
		a_out = tf.Print(a, ['a_value:',a])
		#K.eval(a_out)
		
		log_dir="/cephfs/group/file-teg-datamining-wx-dm-intern/tianyihu/AutoAttention"
		file_writer = tf.summary.FileWriter(log_dir + "/metrics")
		#with tf.Session() as sess:
		#	data_numpy = a.eval()
		#	print(data_numpy)
		# summary = tf.Summary(value=[tf.Summary.Value(tag="a", tensor=a)])
		# summary1 = tf.summary.histogram("a", a)
		# file_writer.add_summary(summary1)
		# with tf.Session() as sess:
		# 	summary_out = sess.run(summary1)
		#	file_writer.add_summary(summary_out)
		file_writer.close()
		# mask
		key_masks = tf.sequence_mask(kes_length[:, 0], self.maxlen) # (None, maxlen)
		if self.weight_softmax:
			paddings = tf.ones_like(a) * (-2 ** 32 + 1)
		else:
			paddings = tf.zeros_like(a)
		a = tf.where(key_masks, a, paddings)

		a = a / (self.emb_dim ** 0.5)

		# softmax
		if self.weight_softmax:
			a = tf.nn.softmax(logits=a)
		outputs = tf.expand_dims(a, axis=1) # (None, 1, maxlen)
		if not self.return_score:
			outputs = tf.matmul(outputs, v) # (None, 1, K)
		# outputs = tf.squeeze(outputs, axis=1) # (None, K)

		return outputs

	def compute_output_shape(self, input_shape):
		return (None, 1, input_shape[0][-1])

	def compute_mask(self, inputs, mask=None):
		return None


def DotProduct(feature_dim_dict, seq_feature_list, embedding_size=8, hist_len_max=16,
				  dnn_use_bn=False, dnn_hidden_units=(200, 80), dnn_activation='sigmoid',
				  l2_reg_dnn=0, l2_reg_embedding=6e-6, dnn_dropout=0, init_std=0.0001, seed=1024, task='binary',
				  weight_softmax=True, is_keys_diff=False):
	"""Instantiates the Deep Interest Network architecture.

	:param feature_dim_dict: dict,to indicate sparse field (**now only support sparse feature**)like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':[]}
	:param seq_feature_list: list,to indicate  sequence sparse field (**now only support sparse feature**),must be a subset of ``feature_dim_dict["sparse"]``
	:param embedding_size: positive integer,sparse feature embedding_size.
	:param hist_len_max: positive int, to indicate the max length of seq input
	:param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
	:param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
	:param dnn_activation: Activation function to use in deep net
	:param l2_reg_dnn: float. L2 regularizer strength applied to DNN
	:param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
	:param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
	:param init_std: float,to use as the initialize std of embedding vector
	:param seed: integer ,to use as random seed.
	:param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
	:param weight_softmax: bool
	:param is_keys_diff: bool
	:return: A Keras model instance.

	"""
	check_feature_config_dict(feature_dim_dict)

	sparse_input, dense_input, user_behavior_input, user_behavior_length = get_input(
		feature_dim_dict, seq_feature_list, hist_len_max)

	sparse_embedding_dict = {feat.name: Embedding(feat.dimension, embedding_size,
												  embeddings_initializer=RandomNormal(
													  mean=0.0, stddev=init_std, seed=seed),
												  embeddings_regularizer=l2(
													  l2_reg_embedding),
												  name='sparse_emb_' + str(i) + '-' + feat.name,
												  mask_zero=(feat.name in seq_feature_list)) for i, feat in
							 enumerate(feature_dim_dict["sparse"])}

	query_emb_list = get_embedding_vec_list(sparse_embedding_dict, sparse_input, feature_dim_dict['sparse'])

	keys_emb_list = get_embedding_vec_list(sparse_embedding_dict, user_behavior_input, feature_dim_dict['sparse'],
										   seq_feature_list, seq_feature_list)
	keys_emb = Add()(keys_emb_list) # sum # (None, maxlen, K)
	# keys_emb = Average()(keys_emb_list)
	v_emb = Concatenate(axis=-1)(keys_emb_list) # (None, maxlen, 2K)


	deep_input_emb_list = get_embedding_vec_list(sparse_embedding_dict, sparse_input, feature_dim_dict['sparse'],
												 mask_feat_list=seq_feature_list)

	query_emb = Lambda(lambda x: tf.concat(x, axis=1))(query_emb_list) # (None, m, K)

	hist = DotProduct_Layer(hist_len_max, embedding_size, weight_softmax, is_keys_diff)([query_emb, keys_emb, v_emb, user_behavior_length])

	deep_input_emb = concat_fun(deep_input_emb_list)

	deep_input_emb = Concatenate()([NoMask()(deep_input_emb), hist])
	deep_input_emb = Flatten()(deep_input_emb)
	if len(dense_input) > 0:
		deep_input_emb = Concatenate()([deep_input_emb] + list(dense_input.values()))

	output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn,
				 dnn_dropout, dnn_use_bn, seed)(deep_input_emb)
	final_logit = Dense(1, use_bias=False)(output)

	output = PredictionLayer(task)(final_logit)
	model_input_list = get_inputs_list([sparse_input, dense_input, user_behavior_input])
	model_input_list += [user_behavior_length]

	model = Model(inputs=model_input_list, outputs=output)
	return model

def AutoAttention(feature_dim_dict, seq_feature_list, embedding_size=8, hist_len_max=16,
		 dnn_use_bn=False, dnn_hidden_units=(200, 80), dnn_activation='sigmoid',
		 l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001, seed=1024, task='binary',
		 weight_softmax=True, is_keys_diff=False):
	"""Instantiates the Deep Interest Network architecture.

	:param feature_dim_dict: dict,to indicate sparse field (**now only support sparse feature**)like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':[]}
	:param seq_feature_list: list,to indicate  sequence sparse field (**now only support sparse feature**),must be a subset of ``feature_dim_dict["sparse"]``
	:param embedding_size: positive integer,sparse feature embedding_size.
	:param hist_len_max: positive int, to indicate the max length of seq input
	:param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
	:param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
	:param dnn_activation: Activation function to use in deep net
	:param l2_reg_dnn: float. L2 regularizer strength applied to DNN
	:param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
	:param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
	:param init_std: float,to use as the initialize std of embedding vector
	:param seed: integer ,to use as random seed.
	:param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
	:param weight_softmax: bool
	:param is_keys_diff: bool
	:return: A Keras model instance.

	"""
	check_feature_config_dict(feature_dim_dict)

	sparse_input, dense_input, user_behavior_input, user_behavior_length = get_input(
		feature_dim_dict, seq_feature_list, hist_len_max)

	sparse_embedding_dict = {feat.name: Embedding(feat.dimension, embedding_size,
												  embeddings_initializer=RandomNormal(
													  mean=0.0, stddev=init_std, seed=seed),
												  embeddings_regularizer=l2(
													  l2_reg_embedding),
												  name='sparse_emb_' + str(i) + '-' + feat.name,
												  mask_zero=(feat.name in seq_feature_list)) for i, feat in
							 enumerate(feature_dim_dict["sparse"])}

	query_emb_list = get_embedding_vec_list(sparse_embedding_dict, sparse_input, feature_dim_dict['sparse'])

	keys_emb_list = get_embedding_vec_list(sparse_embedding_dict, user_behavior_input, feature_dim_dict['sparse'],
									   seq_feature_list, seq_feature_list)
	keys_emb = Add()(keys_emb_list) # sum # (None, maxlen, K)
	# keys_emb = Average()(keys_emb_list)
	v_emb = Concatenate(axis=-1)(keys_emb_list) # (None, maxlen, 2K)


	deep_input_emb_list = get_embedding_vec_list(sparse_embedding_dict, sparse_input, feature_dim_dict['sparse'],
												 mask_feat_list=seq_feature_list)

	query_emb = Lambda(lambda x: tf.concat(x, axis=1))(query_emb_list) # (None, m, K)

	hist = AutoAttention_Layer(hist_len_max, embedding_size, weight_softmax, is_keys_diff)([query_emb, keys_emb, v_emb, user_behavior_length])
	
	#with tf.Session():
	#	print(a_out.eval())
	deep_input_emb = concat_fun(deep_input_emb_list)

	deep_input_emb = Concatenate()([NoMask()(deep_input_emb), hist])
	deep_input_emb = Flatten()(deep_input_emb)
	if len(dense_input) > 0:
		deep_input_emb = Concatenate()([deep_input_emb] + list(dense_input.values()))

	output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn,
				 dnn_dropout, dnn_use_bn, seed)(deep_input_emb)
	final_logit = Dense(1, use_bias=False)(output)

	output = PredictionLayer(task)(final_logit)
	model_input_list = get_inputs_list([sparse_input, dense_input, user_behavior_input])
	model_input_list += [user_behavior_length]

	model = Model(inputs=model_input_list, outputs=output)
	return model
