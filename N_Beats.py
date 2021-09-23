#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Paper : N-BEATS (Neual basis expansion analysis for interpretable time series forecasting)
# https://arxiv.org/abs/1905.10437
# 단변량 시계열에 대한 모델



import tensorflow as tf
from tensorflow import keras

tf.config.experimental_run_functions_eagerly(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True) 
        tf.config.experimental.set_memory_growth(gpus[1], True) 
    except RuntimeError as e:
        print(e)


# In[2]:


class FCStack(keras.layers.Layer) :
    def __init__(self,
                 n_layers = 4,
                 kernel_initializer = "glorot_normal",
                 bias_initializer = "glorot_normal",
                 activation = "relu", 
                 **kwargs) :
        
        self.layers = [keras.layers.Dense(256,
                                  activation = "relu",
                                  kernel_initializer = kernel_initializer,
                                  bias_initializer = bias_initializer) for i in range(n_layers)]
        
        super(FCStack, self).__init__()
    
    def call(self, x) :
        for layer in self.layers :
            x = layer(x)
        return x    


# In[6]:


class BasicBlock(keras.layers.Layer) :
    def __init__(self,
                 pred_point,
                 n_layers = 4,
                 kernel_initializer = "glorot_normal",
                 bias_initializer = "glorot_normal",
                 activation = "relu",
                 **kwargs) :
        
        self.stack = FCStack(n_layers)
        
        self.back_fc = keras.layers.Dense(20)
        self.backcast = keras.layers.Dense(pred_point * 3)
        
        self.fore_fc = keras.layers.Dense(20)
        self.forecast = keras.layers.Dense(pred_point)
        
        super(BasicBlock, self).__init__()
    
    def call(self, x) :
        x = self.stack(x)
        
        back = self.back_fc(x)
        back = self.backcast(back)
        
        fore = self.fore_fc(x)
        fore = self.forecast(fore)
        
        return [back, fore]


# In[15]:


class StackBlock(keras.layers.Layer) :
    def __init__(self,
                 pred_point,
                 n_layers = 4,
                 n_block = 10,
                 return_foreInputs = True,
                 kernel_initializer = "glorot_normal",
                 bias_initializer = "glorot_normal",
                 activation = "relu",
                 **kwargs) :
        
        self.return_fore = return_foreInputs
        self.embedding = keras.layers.Dense(pred_point * 3, kernel_initializer = "glorot_normal", use_bias = False)
        self.blocks = [BasicBlock(pred_point = pred_point,
                                  n_layers = n_layers,
                                  kernel_initializer = kernel_initializer,
                                  bias_initializer = bias_initializer,
                                  activation = activation) for i in range(n_block)]
        
        super(StackBlock, self).__init__()
        
    def call(self, x) :
        inputs = self.embedding(x)
        foreInputs = self.embedding(x)
        forecasts = []
        for block in self.blocks :
            outs = block(inputs)
            inputs, forecast = outs[0], outs[1]
            inputs -= foreInputs
            foreInputs = inputs
            
            forecasts.append(forecast)
            
        if self.return_fore :
            return [foreInputs, sum(forecasts)]

        else :
            return sum(forecasts)
