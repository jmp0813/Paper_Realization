#!/usr/bin/env python
# coding: utf-8

# Attention is all you need의 Transformer입니다.
# https://arxiv.org/abs/1706.03762
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 입력 순서를 고려하지 못하기 때문에 PositionalEncoding을 이용해 순서를 배정합니다.
class PositionalEncoding(keras.layers.Layer) :
    def __init__(self, time_length, n_feature, dtype = tf.float32, **kwargs) :
        super().__init__(dtype = dtype, **kwargs)
        p, i = np.meshgrid(np.arange(time_length), np.arange(n_feature // 2))
        position_embedding = np.empty((1, time_length, n_feature))
        position_embedding[0, :, ::2] = np.sin(p / 10000 ** (2 * i / n_feature)).T
        position_embedding[0, :, 1::2] = np.cos(p / 10000 ** (2 * i / n_feature)).T
        self.position_embedding = tf.constant(position_embedding.astype(self.dtype))
        
    def call(self, inputs) :
        shape = tf.shape(inputs)
        return inputs + self.position_embedding[:, :shape[-2], :shape[-1]]
    
# 일반적인 어텐션을 head 수만큼 늘린 구조입니다.
class MultiHeadAttention(keras.layers.Layer) :
    def __init__(self, n_heads, n_units) :
        self.n_heads = n_heads
        self.n_units = n_units
        super(MultiHeadAttention, self).__init__()
        
    def build(self, x) :
        Q = x[0]
        K = x[1]
        V = x[2]
        
        self.query_weights = []
        self.key_weights = []
        self.value_weights = []
        
        for head in range(self.n_heads) :
            self.query_weights.append(self.add_weight('q_w' + str(head), (Q[-1], self.n_units),
                                                      initializer = keras.initializers.glorot_normal(),
                                                      trainable = True))
            self.key_weights.append(self.add_weight('k_w' + str(head), (K[-1], self.n_units),
                                                    initializer = keras.initializers.glorot_normal(),
                                                    trainable = True))
            self.value_weights.append(self.add_weight("v_w" + str(head), (V[-1], V[-1]),
                                                      initializer = keras.initializers.glorot_normal(),
                                                      trainable = True))
            
        self.linear_weight = self.add_weight('l_w', (self.n_heads * V[-1], V[-1]),
                                             initializer = keras.initializers.glorot_normal(),
                                             trainable = True)
        
        super(MultiHeadAttention, self).build(x)
        
    def call(self, x) :
        Q = x[0]
        K = x[1]
        V = x[2]
        
        Attention = []
        Score = []
        
        for head in range(self.n_heads) :
            Query = keras.backend.dot(Q, self.query_weights[head])
            Key = keras.backend.dot(K, self.key_weights[head])
            Value = keras.backend.dot(V, self.value_weights[head])
            
            Attention.append(tf.matmul(Query, Key, transpose_b = True) / np.sqrt(Q.shape[2]))
            Score.append(keras.backend.softmax(Attention[head]))
            Attention[head] = tf.matmul(Score[head], Value)
        
        output = tf.concat(Attention, axis = 2)
        output = tf.matmul(output, self.linear_weight)
        return output
            
    def compute_output_shape(self, x) :
        return (x[0].shape[-2], x[0].shape[-1])

# Multi-head Attention 뒤에 붙는 단순한 Feed-forward입니다.
class PointWiseFeedForward(keras.layers.Layer) :
    def __init__(self, n_units, **kwargs) :
        self.n_units = n_units
        super(PointWiseFeedForward, self).__init__(**kwargs)
    
    def build(self, input_shape) :
        self.w1 = self.add_weight("w1", (input_shape[2], self.n_units),
                                  initializer = keras.initializers.get("glorot_normal"),
                                  trainable = True)
        self.w2 = self.add_weight("w2", (self.n_units, input_shape[2]),
                                  initializer = keras.initializers.get("glorot_normal"),
                                  trainable = True)
        self.b1 = self.add_weight("b1", (self.n_units, ),
                                  initializer = keras.initializers.get("glorot_normal"),
                                  trainable = True)
        self.b2 = self.add_weight("b2", (input_shape[2], ),
                                  initializer = keras.initializers.get("glorot_normal"),
                                  trainable = True)
        
        super(PointWiseFeedForward, self).build(input_shape)
        
    def call(self, x) :
        dense_1 = keras.backend.dot(x, self.w1) + self.b1
        dense_1 = keras.backend.relu(dense_1)
        dense_2 = keras.backend.dot(dense_1, self.w2) + self.b2
        return dense_2
        
    def compute_output_shape(self, input_shape) :
        return (input_shape[1], input_shape[2])
    
# Transformer의 Encoder 부분입니다.
# 입력과 출력 모두 하나로 이루어집니다.
# Encoder의 출력은 Decoder의 중간 Multi-head attention 모듈의 Query와 Key값으로 넘어갑니다.

class TransformerEncoder(keras.layers.Layer) :
    def __init__(self, n_heads, attn_units, ff_units, dropout, **kwargs) :
        self.MHA = MultiHeadAttention(n_heads, attn_units)
        self.dropout_1 = keras.layers.Dropout(rate = dropout)
        self.Norm_1 = keras.layers.LayerNormalization()
        
        self.PWFF = PointWiseFeedForward(ff_units)
        self.dropout_2 = keras.layers.Dropout(rate = dropout)
        self.Norm_2 = keras.layers.LayerNormalization()
        
        super(TransformerEncoder, self).__init__(**kwargs)
        
    def call(self, x, training) :
        attn = self.MHA([x, x, x])
        attn = self.dropout_1(attn, training = training)
        add_n_norm = self.Norm_1(x + attn)
        
        pwff = self.PWFF(add_n_norm)
        pwff = self.dropout_2(pwff)
        output = self.Norm_2(add_n_norm + pwff)
        return output

# Transformer의 Decoder 부분입니다.
# 입력은 Decoder 자체의 입력과 Encoder의 출력 두 개를 밭습니다.
class TransformerDecoder(keras.layers.Layer) :
    def __init__(self, n_heads, attn_units, ff_units, dropout, **kwargs) :
        self.MHA_1 = MultiHeadAttention(n_heads, attn_units)
        self.dropout_1 = keras.layers.Dropout(rate = dropout)
        self.Norm_1 = keras.layers.LayerNormalization()
        
        self.MHA_2 = MultiHeadAttention(n_heads, attn_units)
        self.dropout_2 = keras.layers.Dropout(rate = dropout)
        self.Norm_2 = keras.layers.LayerNormalization()
        
        self.PWFF = PointWiseFeedForward(ff_units)
        self.dropout_3 = keras.layers.Dropout(rate = dropout)
        self.Norm_3 = keras.layers.LayerNormalization()
        
        super(TransformerDecoder, self).__init__(**kwargs)
        
    def call(self, x, training) :
        origin_input = x[0]
        enc_input = x[1]
        
        attn_1 = self.MHA_1([origin_input, origin_input, origin_input])
        attn_1 = self.dropout_1(attn_1, training = training)
        add_n_norm_1 = self.Norm_1(origin_input + attn_1)
        
        attn_2 = self.MHA_2([enc_input, enc_input, add_n_norm_1])
        attn_2 = self.dropout_2(attn_2, training = training)
        add_n_norm_2 = self.Norm_2(attn_2 + add_n_norm_1)
        
        pwff = self.PWFF(add_n_norm_2)
        pwff = self.dropout_3(pwff)
        output = self.Norm_3(add_n_norm_2 + pwff)
        return output

# Encoder-Decoder 구조를 중첩해서 쌓을 때 다음 구조와의 Residual Connection을 위한 SEBlock입니다.
# Bottle-neck 구조를 띄고있어 정보를 잘 뽑아낼 수 있을거라 기대합니다.
# 실제 하는 역할은 Decoder에서 출력한 결과 벡터들에 가중치를 매기는 것입니다.
# LeNet의 Inception Module과도 유사하게 출력을 시간 방향으로 연결할 수 있을 것 같은데
# 일단은 Bottle neck 역할을 하는 부분은 두 모듈에 모두 있으니 차차 생각해보겠습니다.

class SEBlock(keras.layers.Layer) :
    def __init__(self, reduction_ratio = 16, dense_1_kernel_init = "he_normal", dense_2_kernel_init = "he_normal",
                 dense_1_bias_init = "zeros", dense_2_bias_init = "zeros", **kwargs) :
        """
        reduction_ratio : Bottle neck의 정도를 표현합니다.
        
        SEBlock : 2개의 Dense를 거쳐 입력받은 벡터의 크기와 동일한 출력을 반환합니다.
        """
        self.dense_1_kernel = dense_1_kernel_init
        self.dense_2_kernel = dense_2_kernel_init
        self.dense_1_bias = dense_1_bias_init
        self.dense_2_bias = dense_2_bias_init
        self.reduction_ratio = reduction_ratio
                
        super(SEBlock, self).__init__(**kwargs)
        
    def build(self, input_shape) :
        self.w1 = self.add_weight("w1", (input_shape[2], input_shape[2] // self.reduction_ratio),
                                  initializer = keras.initializers.get(self.dense_1_kernel),
                                  trainable = True)
        self.w2 = self.add_weight("w2", (input_shape[2] // self.reduction_ratio, 1, input_shape[2]),
                                  initializer = keras.initializers.get(self.dense_2_kernel),
                                  trainable = True)
        self.b1 = self.add_weight("b1", (input_shape[2] // self.reduction_ratio),
                                  initializer = keras.initializers.get(self.dense_1_bias),
                                  trainable = True)
        self.b2 = self.add_weight("b2", (input_shape[2]),
                                  initializer = keras.initializers.get(self.dense_2_bias),
                                  trainable = True)
        super(SEBlock, self).build(input_shape)
        
    def call(self, x) :
        pool = keras.backend.mean(x, axis = 1, keepdims = True)
        dense_1 = keras.backend.dot(pool, self.w1) + self.b1
        dense_1 = keras.backend.relu(dense_1)
        dense_2 = keras.backend.dot(dense_1, self.w2) + self.b2
        dense_2 = keras.backend.sigmoid(dense_2)
        return dense_2
    
    def compute_output_shape(self, input_shape) :
        return (input_shape[0], input_shape[2])
    
class TransformerScheduler(keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, attn_units, warmup_steps=4000):
        super(TransformerScheduler, self).__init__()
        self.attn_units = attn_units
        self.attn_units = tf.cast(self.attn_units, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.attn_units) * tf.math.minimum(arg1, arg2)