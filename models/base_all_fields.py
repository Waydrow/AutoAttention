# -*- coding:utf-8 -*-

from collections import OrderedDict
import tensorflow as tf
from deepctr.input_embedding import get_inputs_list, create_singlefeat_inputdict, get_embedding_vec_list
from deepctr.layers.core import DNN, PredictionLayer
from deepctr.layers.sequence import AttentionSequencePoolingLayer
from deepctr.layers.utils import concat_fun, NoMask
from deepctr.utils import check_feature_config_dict
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import Input, Dense, Embedding, Concatenate, Flatten, Lambda, Add, Average, Layer
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import Zeros, glorot_normal
from tensorflow.python.keras import backend as K


class Base_Layer(Layer):

	def __init__(self, att_hidden_units=(80, 40), att_activation='sigmoid', weight_normalization=True,
				 return_score=False,
				 supports_masking=False, **kwargs):

		self.att_hidden_units = att_hidden_units
		self.att_activation = att_activation
		self.weight_normalization = weight_normalization
		self.return_score = return_score
		super(Base_Layer, self).__init__(**kwargs)
		self.supports_masking = supports_masking

	def build(self, input_shape):
		if not self.supports_masking:
			if not isinstance(input_shape, list) or len(input_shape) != 3:
				raise ValueError('A `AttentionSequencePoolingLayer` layer should be called '
								 'on a list of 3 inputs')

			if len(input_shape[0]) != 3 or len(input_shape[1]) != 3 or len(input_shape[2]) != 2:
				raise ValueError(
					"Unexpected inputs dimensions,the 3 tensor dimensions are %d,%d and %d , expect to be 3,3 and 2" % (
						len(input_shape[0]), len(input_shape[1]), len(input_shape[2])))

			if input_shape[0][-1] != input_shape[1][-1] or input_shape[0][1] != 1 or input_shape[2][1] != 1:
				raise ValueError('A `AttentionSequencePoolingLayer` layer requires '
								 'inputs of a 3 tensor with shape (None,1,embedding_size),(None,T,embedding_size) and (None,1)'
								 'Got different shapes: %s' % (input_shape))
		else:
			pass
		self.local_att = Base_LocalActivationUnit(
			self.att_hidden_units, self.att_activation, l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, )
		super(Base_Layer, self).build(
			input_shape)  # Be sure to call this somewhere!

	def call(self, inputs, mask=None, training=None, **kwargs):

		if self.supports_masking:
			if mask is None:
				raise ValueError(
					"When supports_masking=True,input must support masking")
			queries, keys = inputs
			key_masks = tf.expand_dims(mask[-1], axis=1)

		else:

			queries, keys, keys_length = inputs
			hist_len = keys.get_shape()[1]
			key_masks = tf.sequence_mask(keys_length, hist_len)

		attention_score = self.local_att([queries, keys], training=training)

		outputs = tf.transpose(attention_score, (0, 2, 1))

		if self.weight_normalization:
			paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
		else:
			paddings = tf.zeros_like(outputs)

		outputs = tf.where(key_masks, outputs, paddings)

		if self.weight_normalization:
			outputs = tf.nn.softmax(outputs)

		if not self.return_score:
			outputs = tf.matmul(outputs, keys)

		outputs._uses_learning_phase = attention_score._uses_learning_phase

		return outputs

	def compute_output_shape(self, input_shape):
		if self.return_score:
			return (None, 1, input_shape[1][1])
		else:
			return (None, 1, input_shape[0][-1])

	def compute_mask(self, inputs, mask):
		return None

	def get_config(self, ):

		config = {'att_hidden_units': self.att_hidden_units, 'att_activation': self.att_activation,
				  'weight_normalization': self.weight_normalization, 'return_score': self.return_score,
				  'supports_masking': self.supports_masking}
		base_config = super(Base_Layer, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class Base_LocalActivationUnit(Layer):
	def __init__(self, hidden_units=(64, 32), activation='sigmoid', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024,
				 **kwargs):
		self.hidden_units = hidden_units
		self.activation = activation
		self.l2_reg = l2_reg
		self.dropout_rate = dropout_rate
		self.use_bn = use_bn
		self.seed = seed
		super(Base_LocalActivationUnit, self).__init__(**kwargs)
		self.supports_masking = True

	def build(self, input_shape):

		if not isinstance(input_shape, list) or len(input_shape) != 2:
			raise ValueError('A `LocalActivationUnit` layer should be called '
							 'on a list of 2 inputs')

		if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
			raise ValueError("Unexpected inputs dimensions %d and %d, expect to be 3 dimensions" % (
				len(input_shape[0]), len(input_shape[1])))

		# if input_shape[0][-1] != input_shape[1][-1] or input_shape[0][1] != 1:
		# 	raise ValueError('A `LocalActivationUnit` layer requires '
		# 					 'inputs of a two inputs with shape (None,1,embedding_size) and (None,T,embedding_size)'
		# 					 'Got different shapes: %s,%s' % (input_shape))
		size = 4 * \
			   int(input_shape[0][-1]
				   ) if len(self.hidden_units) == 0 else self.hidden_units[-1]
		self.kernel = self.add_weight(shape=(size, 1),
									  initializer=glorot_normal(
										  seed=self.seed),
									  name="kernel")
		self.bias = self.add_weight(
			shape=(1,), initializer=Zeros(), name="bias")
		self.dnn = DNN(self.hidden_units, self.activation, self.l2_reg,
					   self.dropout_rate, self.use_bn, seed=self.seed)

		self.dense = tf.keras.layers.Lambda(lambda x:tf.nn.bias_add(tf.tensordot(
			x[0], x[1], axes=(-1, 0)), x[2]))

		super(Base_LocalActivationUnit, self).build(
			input_shape)  # Be sure to call this somewhere!

	def call(self, inputs, training=None, **kwargs):

		query, keys = inputs

		keys_len = keys.get_shape()[1]
		queries = K.repeat_elements(query, keys_len, 1)

		# att_input = tf.concat(
		# 	[queries, keys, queries - keys, queries * keys], axis=-1)
		att_input = tf.concat([queries, keys], axis=-1)

		att_out = self.dnn(att_input, training=training)

		attention_score = self.dense([att_out,self.kernel,self.bias])

		return attention_score

	def compute_output_shape(self, input_shape):
		return input_shape[1][:2] + (1,)

	def compute_mask(self, inputs, mask):
		return mask

	def get_config(self, ):
		config = {'activation': self.activation, 'hidden_units': self.hidden_units,
				  'l2_reg': self.l2_reg, 'dropout_rate': self.dropout_rate, 'use_bn': self.use_bn, 'seed': self.seed}
		base_config = super(Base_LocalActivationUnit, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

def get_input(feature_dim_dict, seq_feature_list, seq_max_len):
	sparse_input, dense_input = create_singlefeat_inputdict(feature_dim_dict)
	user_behavior_input = OrderedDict()
	for i, feat in enumerate(seq_feature_list):
		user_behavior_input[feat] = Input(shape=(seq_max_len,), name='seq_' + str(i) + '-' + feat)

	return sparse_input, dense_input, user_behavior_input


def Base_All_Fields(feature_dim_dict, seq_feature_list, embedding_size=8, hist_len_max=16,
		 dnn_use_bn=False, dnn_hidden_units=(200, 80), dnn_activation='sigmoid', flag='add',
		 l2_reg_dnn=0, l2_reg_embedding=3e-6, dnn_dropout=0, init_std=0.0001, seed=1024, task='binary'):
	"""Instantiates the Deep Interest Network architecture.

	:param feature_dim_dict: dict,to indicate sparse field (**now only support sparse feature**)like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':[]}
	:param seq_feature_list: list,to indicate  sequence sparse field (**now only support sparse feature**),must be a subset of ``feature_dim_dict["sparse"]``
	:param embedding_size: positive integer,sparse feature embedding_size.
	:param hist_len_max: positive int, to indicate the max length of seq input
	:param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
	:param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
	:param dnn_activation: Activation function to use in deep net
	:param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
	:param att_activation: Activation function to use in attention net
	:param att_weight_normalization: bool.Whether normalize the attention score of local activation unit.
	:param l2_reg_dnn: float. L2 regularizer strength applied to DNN
	:param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
	:param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
	:param init_std: float,to use as the initialize std of embedding vector
	:param seed: integer ,to use as random seed.
	:param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
	:return: A Keras model instance.

	"""
	check_feature_config_dict(feature_dim_dict)

	sparse_input, dense_input, user_behavior_input = get_input(
		feature_dim_dict, seq_feature_list, hist_len_max)

	sparse_embedding_dict = {feat.name: Embedding(feat.dimension, embedding_size,
												  embeddings_initializer=RandomNormal(
													  mean=0.0, stddev=init_std, seed=seed),
												  embeddings_regularizer=l2(
													  l2_reg_embedding),
												  name='sparse_emb_' + str(i) + '-' + feat.name,
												  mask_zero=(feat.name in seq_feature_list)) for i, feat in
							 enumerate(feature_dim_dict["sparse"])}

	# query_emb_list = get_embedding_vec_list(sparse_embedding_dict, sparse_input, feature_dim_dict['sparse'],
	#                                         seq_feature_list, seq_feature_list)
	query_emb_list = get_embedding_vec_list(sparse_embedding_dict, sparse_input, feature_dim_dict['sparse'])
	keys_emb_list = get_embedding_vec_list(sparse_embedding_dict, user_behavior_input, feature_dim_dict['sparse'],
										seq_feature_list, seq_feature_list)

	deep_input_emb_list = get_embedding_vec_list(sparse_embedding_dict, sparse_input, feature_dim_dict['sparse'],
												 mask_feat_list=seq_feature_list)

	if flag == 'add':
		query_emb = Add()(query_emb_list) # (None, 1, K)
	else:
		query_emb = concat_fun(query_emb_list) # (None, 1, mK)
	keys_emb = Add()(keys_emb_list) # (None, maxlen, K)
	v_emb = concat_fun(keys_emb_list)

	deep_input_emb = concat_fun(deep_input_emb_list)

	# sum pooling
	# hist = Lambda(lambda x: tf.reduce_sum(x, axis=1, keep_dims=True))(keys_emb)
	hist = Base_Layer(supports_masking=True)([query_emb, keys_emb])
	# Don't need mask, since the previous layer (Embedding) will pass this parameter through mask_zero arguments


	deep_input_emb = Concatenate()([NoMask()(deep_input_emb), hist])
	deep_input_emb = Flatten()(deep_input_emb)
	if len(dense_input) > 0:
		deep_input_emb = Concatenate()([deep_input_emb] + list(dense_input.values()))

	output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn,
				 dnn_dropout, dnn_use_bn, seed)(deep_input_emb)
	final_logit = Dense(1, use_bias=False)(output)

	output = PredictionLayer(task)(final_logit)
	model_input_list = get_inputs_list([sparse_input, dense_input, user_behavior_input])

	model = Model(inputs=model_input_list, outputs=output)
	return model
