#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import sys
from tensorflow import keras

sys.path.append("/home/jovyan/JM/Models")

from Transformer import MultiHeadAttention


# https://arxiv.org/pdf/1912.09363.pdf
# Temporal Fusion Transformer의 구현 코드입니다.

# 모델의 복잡도는 최초 Input의 embedding size를 늘리거나 Variable selection에서 GRN을 더 쌓으면 됩니다.

# 모델의 가장 중요한 매커니즘인 Gating Linear Unit과 Variable Selection은 다른 방법에도 응용 가능할 것으로 보입니다.

# 이름에 Transformer가 들어가있지만 실제 Transformer와 같은 점은 아래 두 가지가 전부인 것으로 이해했습니다.
#  1. Encoder-Decoder 구조
#  2. Multi-head self-attention 사용



# In[3]:


class GLU(keras.layers.Layer) :
    """
    Gated Linear Unit
    GRN에 사용될 Unit입니다. 본 모델에서 Gate 역할을 수행합니다.
    하나의 Input을 받고 두 개의 perceptron를 거쳐 Element-wise product를 취하고 반환합니다.
    첫 번째 Dense : Sigmoid activation을 취한 perceptron (Bias 사용)
    두 번째 Dense : Perceptron을 취하고 linear하게 반환 (Bias 사용)
    
    동일한 입력에 대해 두 개의 서로 다른 weight를 거치고 이의 element-wise product 결과를 반환합니다.
    
    인자는 일반 perceptron에 들어가는 것들과 같습니다. 
    """
    def __init__(self,
                 n_units,
                 kernel_initializer = "glorot_normal",
                 bias_initializer = "glorot_normal", **kwargs) :
        
        self.dense_1 = keras.layers.Dense(n_units,
                                          kernel_initializer = kernel_initializer,
                                          bias_initializer = bias_initializer,
                                          activation = "sigmoid")
        self.dense_2 = keras.layers.Dense(n_units,
                                          kernel_initializer = kernel_initializer,
                                          bias_initializer = bias_initializer,
                                          activation = None)
        super(GLU, self).__init__()
        
    def call(self, x) :
        d_1 = self.dense_1(x)
        d_2 = self.dense_2(x)
        return tf.multiply(d_1, d_2)


# In[4]:


class Gated_Add_n_Norm(keras.layers.Layer) :
    """
    Skip-connection을 위한 구조입니다.
    입력받은 두 값을 서로 더하고 Layer Normalization을 취합니다.
    첫 번째 인자로 Original Input, 두 번째 인자로 처리를 거친 Input을 넣습니다
    (y = x + f(x)에서 Original Input이 x, 처리된 Input이 f(x))    
    """
    def __init__(self, n_units,
                 kernel_initializer = "glorot_normal",
                 bias_initializer = "glorot_normal",
                 activation = None, **kwargs) :
        self.glu = GLU(n_units,
                       kernel_initializer = kernel_initializer,
                       bias_initializer = bias_initializer)
        
        self.projection = keras.layers.Dense(n_units,
                                             kernel_initializer = kernel_initializer,
                                             bias_initializer = bias_initializer,
                                             activation = activation)
        super(Gated_Add_n_Norm, self).__init__()
        
    def call(self, x) :
        original_input = x[0]
        processed_input = x[1]
        
        if x[0].shape[-1] != x[1].shape[-1] :
            original_input = self.projection(original_input)
        
        processed_input = self.glu(processed_input)
        
        return keras.layers.LayerNormalization()(original_input + processed_input)


# In[5]:


class GRN(keras.layers.Layer) :
    """
    TFT에서 기본 구성요소 중 하나인 Unit입니다.
    하나에서 두 개의 입력을 받고 처리합니다.
    
    입력 1 : Primary input
    입력 2 : (Optional) External Context (Context vector, Time-invariant한 입력으로, 저희 case의 경우 coin_index가 될 수 있습니다.) 
    
    Skip-connection입니다.
    Primary input은 2개의 perceptron과 GLU를 거쳐 원래 Primary Input과 더해진 후 반환됩니다.
    Extternal Context를 입력으로 받은 경우 perceptron에서 같이 처리합니다.
    
    만일 두 번째 인자를 받지 않는다면 단순히 첫 번째 인자에 대해서만 Skip-connection을 수행합니다.
    """
    def __init__(self,
                 n_units,
                 dropout = None,
                 kernel_initializer = "glorot_normal",
                 bias_initializer = "glorot_normal",
                 use_exo_var = False, **kwargs) :
        
        self.addNorm = Gated_Add_n_Norm(n_units)
        
        self.dense_1 = keras.layers.Dense(n_units,
                                          kernel_initializer = kernel_initializer,
                                          bias_initializer = bias_initializer,
                                          activation = None)
        
        self.dense_2 = keras.layers.Dense(n_units,
                                          kernel_initializer = kernel_initializer,
                                          bias_initializer = bias_initializer,
                                          activation = None)
        
        self.dense_3 = None
        if use_exo_var :
            self.dense_3 = keras.layers.Dense(n_units,
                                              kernel_initializer = kernel_initializer,
                                              use_bias = False)
        
        self.dropout = None
        if dropout is not None :
            self.dropout = keras.layers.Dropout(dropout)
        
        super(GRN, self).__init__()
        
    def call(self, x) :
        if self.dense_3 is not None :
            linear_vars = x[0]
            time_indep_vars = x[1]
            
            d_2 = self.dense_2(linear_vars)
            d_3 = self.dense_3(time_indep_vars)
            
            if len(d_3.shape) < 3 :
                d_3 = tf.expand_dims(d_3, 1)
            h_2 = keras.activations.elu(d_2 + d_3)

        else :
            linear_vars = x
        
            d_2 = self.dense_2(linear_vars)
            
            h_2 = keras.activations.elu(d_2)
            
        h_1 = self.dense_1(h_2)
        if self.dropout is not None :
            h_1 = self.dropout(h_1)
        
        
        return self.addNorm([linear_vars, h_1])


# In[6]:


class InputsToEmbedding(keras.layers.Layer) :
    """
    Input의 각 feature별로 embedding을 수행합니다.
    저희의 경우 coin_index라는 time-invariant한 categorical feature와, 이 외의 time-variant한 continuous feature로 구성되어 있습니다.
    따라서 feature의 type 별로 서로 다른 작업을 수행합니다.
    
    1) Categorical feature
        Input feature의 column별로 embedding을 수행합니다.
        (각 column마다 서로 다른 weight를 적용시켜 dimension 확장)
    
    2) Continuous feature
        Input feature의 column별로 linear transformation을 수행합니다.
        (각 column마다 서로 다른 weight를 적용시켜 dimension 확장)
        
    ex) 80 X 6의 matrix에 대해 80 X 1 크기의 vector마다 전부 서로 다른 weight를 거칩니다.
        weight size가 1 X 10이라면 이 결과로 80 X 6 X 10 크기의 결과가 출력됩니다.
        (본 module에선 최종 결과가 concatenate되어 출력됩니다.)
    
    n_embedding : Continuous feature의 linear embedding dimension 크기
    n_features : feature의 수
    mode : categorical feature일 경우 cat, continuous feature일 경우 cont
    kernel, bias_initializer : 가중치 초기화 방법
    activation : 활성화 함수. String으로 인자 전달
    use_bias : linear embedding dense의 bias 사용 여부
    """
    def __init__(self,
                 n_embedding,
                 n_features,
                 mode,
                 kernel_initializer = "glorot_normal",
                 bias_initializer = "glorot_normal",
                 activation = None,
                 use_bias = False,
                 **kwargs) :
        
        
        self.embedding_weights = []
        
        if mode == "cont" :
            for i in range(n_features) :
                to_embedding = keras.layers.TimeDistributed(
                                keras.layers.Dense(n_embedding,
                                                   kernel_initializer = kernel_initializer,
                                                   bias_initializer = bias_initializer,
                                                   use_bias = use_bias,
                                                   activation = activation)
                                )

                self.embedding_weights.append(to_embedding)
        
        elif mode == "cat" :
            for i in range(n_features) :
                to_embedding = keras.layers.Embedding(input_dim = n_features, output_dim = n_embedding) 
                self.embedding_weights.append(to_embedding)
        
        else :
            raise Exception("mode 인자는 cont 또는 cat 둘 중 하나만 가능합니다.")
        
        super(InputsToEmbedding, self).__init__()
        
    def call(self, x) :
        outputs = []
        for i in range(len(self.embedding_weights)) :
            embedding_result = self.embedding_weights[i](tf.expand_dims(x[..., i], -1))
            if 1 in embedding_result.shape :
                embedding_result = tf.squeeze(embedding_result, axis = 1)
            outputs.append(embedding_result)
        
        return outputs


# In[7]:


class VariableSelection(keras.layers.Layer) :
    """
    예측에 적합한 변수를 선택해주는 층입니다.
    
    n_units : GRN의 unit 수
    n_features : Input feature의 수
    n_embedding : Continuous feature의 linear embedding dimension 크기
    mode : Continuous feature일 경우 cont, Categorical feature일 경우 cat
    kernel, bias_initializer : 가중치 초기화 방법
    activation : 활성화 함수. String으로 인자 전달
    use_bias : linear embedding dense의 bias 사용 여부

    """
    def __init__(self,
                 n_units,
                 n_features,
                 n_embedding, 
                 mode,
                 kernel_initializer = "glorot_normal",
                 bias_initializer = "glorot_normal",
                 dropout = None,
                 use_bias = False, 
                 activation = None,
                 use_exo_var = False, **kwargs) :
        
        
        self.embedding = InputsToEmbedding(n_embedding,
                                           n_features,
                                           mode,
                                           kernel_initializer,
                                           bias_initializer,
                                           activation,
                                           use_bias)
        self.grns = []
        for i in range(n_features) :
            grn = GRN(n_units, dropout = dropout, use_exo_var = False)
            self.grns.append(grn)
        
        self.flatten_grn = GRN(n_units, dropout = dropout, use_exo_var = use_exo_var)
        
        self.use_exo_var = use_exo_var
        self.mode = mode
        super(VariableSelection, self).__init__()
        
    def call(self, x) :
        if self.use_exo_var :
            major_vars = x[0]
            external_vars = x[1]
        else :
            major_vars = x
            
        embedded = self.embedding(major_vars)
        flatten = keras.layers.Concatenate()(embedded)
        
        if self.use_exo_var :
            variable_weights = self.flatten_grn([flatten, external_vars])
        else :
            variable_weights = self.flatten_grn(flatten)
        
        variable_weights = keras.backend.softmax(variable_weights)
        
        grn_results = []
        for i in range(len(self.grns)) :
            result = self.grns[i](embedded[i])
            result = tf.multiply(result, tf.expand_dims(variable_weights[..., i], axis = -1))
            grn_results.append(result)
            
        grns_concat = tf.stack(grn_results, axis = -1)
        
        out = tf.reduce_sum(grns_concat, axis = -1)
        
        ##
        # 수정 필요
#        if self.mode == "cat" :
#            out = keras.layers.Permute((2, 1))(out)
        ##
        return out
        


# In[8]:


class  StaticCovariateEncoder(keras.layers.Layer) :
    """
    이미 알고있는 feature를 해석해 그 결과를 Static enrichment, LSTM Encoder의 initial state, Variable Selection에 활용합니다.
    저희 경우엔 coin_index에 해당됩니다.
    
    입력값을 서로 다른 layer에서 활용하기 위해 각 활용 layer별로 서로 다른 weight를 적용합니다.
    인자는 GRN과 같습니다.
    """
    def __init__(self,
                 n_units,
                 kernel_initializer = "glorot_normal",
                 bias_initializer = "glorot_normal",
                 dropout = None,
                 **kwargs) :
        self.s_w = GRN(n_units = n_units,
                       kernel_initializer = kernel_initializer,
                       bias_initializer = bias_initializer,
                       dropout = dropout)
        self.e_w = GRN(n_units = n_units,
                       kernel_initializer = kernel_initializer,
                       bias_initializer = bias_initializer,
                       dropout = dropout)
        self.c_w = GRN(n_units = n_units,
                       kernel_initializer = kernel_initializer,
                       bias_initializer = bias_initializer,
                       dropout = dropout)
        self.h_w = GRN(n_units = n_units,
                       kernel_initializer = kernel_initializer,
                       bias_initializer = bias_initializer,
                       dropout = dropout)
        
        super(StaticCovariateEncoder, self).__init__()
        
    def call(self, x) :
        c_s = self.s_w(x)
        c_e = self.e_w(x)
        c_c = self.c_w(x)
        c_h = self.h_w(x)
        
        return [c_s, c_e, c_c, c_h]


# In[9]:


class StaticEnrichment(keras.layers.Layer) :
    """
    Temporal Fusion Decoder의 첫 module입니다.
    단순히 GRN으로만 구성되어있습니다.
    Static Covariate Encoder와 Gate를 거친 LSTM Encoder의 결과를 입력으로 받습니다.
    
    인자는 GRN과 같습니다.
    """
    def __init__(self,
                 n_units,
                 kernel_initializer = "glorot_normal",
                 bias_initializer = "glorot_normal",
                 dropout = None,
                 use_exo_var = False,
                 **kwargs) :
        self.grn = GRN(n_units = n_units,
                       kernel_initializer = kernel_initializer,
                       bias_initializer = bias_initializer,
                       dropout = dropout,
                       use_exo_var = use_exo_var)
        
        super(StaticEnrichment, self).__init__()
    def call(self, x) :
        return self.grn(x)


# In[10]:


class TemporalSelfAttention(keras.layers.Layer) :
    """
    Scaled-dot product 기반의 self-attention을 여러개 겹친 multi-head attention입니다.
    최적화가 제대로 되지 않은 듯 해 메모리 리소스를 꽤 많이 먹는 것 같습니다... 차차 수정하겠습니다.
    
    입력으로 Static Enrichment의 결과를 받아 이를 self-attention합니다.
    따라서 자연어 처리와 달리 Query, Key, Value 모두 같은 값입니다.
    
    Self-attention의 결과를 Gate에 통과시켜 출력합니다.
    
    인자로는 Gate module의 인자와 Multi-head attention의 두 인자가 들어갑니다.
    Multi-head attention : n_attn_head : head의 수를 결정합니다. (지금 상황에서 리소스를 잡아먹는 주범입니다.)
                           attn_unit : head 하나에서 사용할 unit의 수 입니다.
    """
    def __init__(self,
                 n_attn_head,
                 attn_unit,
                 gate_units, 
                 gate_kernel_initializer = "glorot_normal",
                 gate_bias_initializer = "glorot_normal",
                 gate_activation = None,
                 **kwargs) :
        self.multi_head_attention = MultiHeadAttention(n_heads = n_attn_head, n_units = attn_unit)
        
        self.gated_add_norm = Gated_Add_n_Norm(n_units = gate_units,
                                               kernel_initializer = gate_kernel_initializer,
                                               bias_initializer = gate_bias_initializer,
                                               activation = gate_activation)
        
        super(TemporalSelfAttention, self).__init__()
        
    def call(self, x) :
        self_attention = self.multi_head_attention([x, x, x])
        return self.gated_add_norm([x, self_attention])


# In[11]:


class PointwiseFeedForward(keras.layers.Layer) :
    """
    Temporal self-attention의 결과를 받아 정보를 종합합니다.
    GRN과 Gate를 거쳐 결과를 출력합니다.
    
    해당 Module은 Output 직전에 위치힙나다.
    """
    def __init__(self,
                 n_units,
                 gate_units,
                 kernel_initializer = "glorot_normal",
                 bias_initializer = "glorot_normal",
                 dropout = None,
                 activation = None,
                 **kwargs) :
        self.grn = GRN(n_units = n_units,
                       kernel_initializer = kernel_initializer,
                       bias_initializer = bias_initializer,
                       dropout = dropout,
                       use_exo_var = False)
        
        self.gate_add_norm = Gated_Add_n_Norm(n_units = gate_units,
                                              kernel_initializer = kernel_initializer,
                                              bias_initializer = bias_initializer,
                                              activation = activation)

        super(PointwiseFeedForward, self).__init__()
        
    def call(self, x) :
        attn_out = x[0]
        gate_lstm_out = x[1]
        attn_out = self.grn(attn_out)
        return self.gate_add_norm([attn_out, gate_lstm_out])


# In[12]:


def default_tft_model(n_static, n_time, n_units, n_attn_head, attn_unit, gate_units) :
    static_inputs = keras.layers.Input(shape = (n_static))
    static_vs = VariableSelection(n_units = n_units, n_features = n_static, n_embedding = 10, mode = "cat")(static_inputs)
    c_s, c_e, c_c, c_h = StaticCovariateEncoder(n_units = n_units)(static_vs)

    time_inputs = keras.layers.Input(shape = (120, n_time))
    time_vs = VariableSelection(n_units = n_units,
                                n_features = n_time,
                                n_embedding = 20,
                                mode = "cont",
                                use_exo_var = True)([time_inputs, c_s])
    
    lstm_initial_state = [c_h, c_c]

    enc_lstm = keras.layers.LSTM(units = n_units,
                                 recurrent_initializer = "glorot_normal",
                                 return_sequences = True)(time_vs, initial_state = lstm_initial_state)

    gate = Gated_Add_n_Norm(n_units = n_units)([enc_lstm, time_vs])

    enrich = StaticEnrichment(n_units, use_exo_var = True)([gate, c_e])

    attn = TemporalSelfAttention(n_attn_head = n_attn_head, attn_unit = attn_unit, gate_units = gate_units)(enrich)

    ff = PointwiseFeedForward(n_units = n_units, gate_units = gate_units)([attn, gate])

    OutputLayer = keras.layers.TimeDistributed(keras.layers.Dense(1, activation = None))(ff)
    
    default_model = keras.models.Model(inputs = [static_inputs, time_inputs], outputs = OutputLayer)
    return default_model


# In[ ]:




