#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import sys
import os

from tensorflow import keras


# In[18]:


class Encoder(keras.layers.Layer) :
    """
    Seq2Seq의 Encoder layer입니다.
    Encoder의 n_layer와 n_units는 Decoder에서도 동일한 값으로 사용해야 합니다.
    
    n_layer = Encoding의 RNN layer 수
    n_units = RNN 당 unit 수
    use_skip_connection : Output = original_input + RNN(original_input)
    
    모든 RNN layer에 대한 hidden states와 cell states를 반환합니다.
    (hidden states.shape == n_layers * (batch_size, n_units),
     cell_states.shape == n_layers * (batch_size, n_units))
    """
    def __init__(self,
                 n_layer,
                 n_units,
                 use_skip_connection = True,
                 kernel_initializer = "glorot_normal",
                 bias_initializer = "glorot_normal", 
                 recurrent_initializer = "glorot_normal",
                 **kwargs) :
        
        self.enc_layers = [keras.layers.LSTM(n_units,
                                             kernel_initializer = kernel_initializer,
                                             bias_initializer = bias_initializer,
                                             recurrent_initializer = recurrent_initializer,
                                             return_sequences = True,
                                             return_state = True) for i in range(n_layer)]
        
        self.use_skip = use_skip_connection
        if use_skip_connection :
            self.linear_transform = keras.layers.TimeDistributed(keras.layers.Dense(n_units, use_bias = False))
        
        super(Encoder, self).__init__()
        
    def call(self, x) :
        sequence = x
        
        if self.use_skip :
            sequence = self.linear_transform(sequence)
        
        outputs = []
        hidden_states = []
        cell_states = []
        
        for layer in self.enc_layers :
            if self.use_skip :
                prev_inputs = []
                prev_inputs.append(sequence)
            sequence, hidden, cell = layer(sequence)
            
            if self.use_skip :
                sequence += prev_inputs[-1]
            
            outputs.append(sequence)
            hidden_states.append(hidden)
            cell_states.append(cell)
            
        return [hidden_states, cell_states]


# In[22]:


class Decoder(keras.layers.Layer) :
    """
    Seq2Seq의 Decoder layer입니다.
    Encoder에서 사용한 n_layer와 n_units와 같은 값을 인자로 받습니다.
    
    n_layer = Encoding의 RNN layer 수
    n_units = RNN 당 unit 수
    use_skip_connection : Output = original_input + RNN(original_input)
    
    Decoder의 마지막 RNN layer의 output을 모든 time step에 대해 반환합니다.
    """
    def __init__(self,
                 n_layer,
                 n_units,
                 use_skip_connection = True,
                 kernel_initializer = "glorot_normal",
                 bias_initializer = "glorot_normal",
                 recurrent_initializer = "glorot_normal",
                 **kwargs) :

        self.dec_layers = [keras.layers.LSTM(n_units,
                                             kernel_initializer = kernel_initializer,
                                             bias_initializer = bias_initializer,
                                             recurrent_initializer = recurrent_initializer,
                                             return_sequences = True) for i in range(n_layer)]
        
        self.use_skip = use_skip_connection
        if use_skip_connection :
            self.linear_transform = keras.layers.TimeDistributed(keras.layers.Dense(n_units, use_bias = False))
        
        super(Decoder, self).__init__()
        
    def call(self, x, encoder_states) :
        sequence = x
        hidden_states = encoder_states[0]
        cell_states = encoder_states[1]
        
        if self.use_skip :
            sequence = self.linear_transform(sequence)
        
        for layer in range(len(self.dec_layers)) :
            if self.use_skip :
                prev_inputs = []
                prev_inputs.append(sequence)
            
            sequence = self.dec_layers[layer](sequence, initial_state = [hidden_states[layer], cell_states[layer]])
            if (self.use_skip) :
                sequence += prev_inputs[-1]
            
        return sequence


# In[23]:


class OutputLayer(keras.layers.Layer) :
    """
    Decoder의 출력을 저희가 필요한 형식으로 변환해주는 Layer입니다.
    단순한 perceptron을 적용시킵니다.
    
    use_time_distributed : 입력으로 받은 sequence에 대해 time-wise하게 dense를 적용할 것인지 결정합니다.
                           False인 경우 time과 무관하게 계산합니다. (출력 형태는 동일합니다.)
                           (Default : True)
    
    n_units : 출력 차원을 결정합니다. 출력 형태는 (None, time_step, n_units) 입니다.
    """
    def __init__(self,
                 n_units = 1,
                 use_time_distributed = True,
                 kernel_initializer = "glorot_normal",
                 bias_initializer = "glorot_normal",
                 **kwargs) :
    
        if use_time_distributed == True :
            self.dense = keras.layers.TimeDistributed(keras.layers.Dense(n_units,
                                                                         kernel_initializer = kernel_initializer,
                                                                         bias_initializer = bias_initializer))
        else :
            self.dense= keras.layers.Dense(n_units,
                                           kernel_initializer = kernel_initializer,
                                           bias_initializer = bias_initializer)

        super(OutputLayer, self).__init__()
    
    def call(self, x) :
        return self.dense(x)

