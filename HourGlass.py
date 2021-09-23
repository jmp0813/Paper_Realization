#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus :
    try :
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e :
        print(e)


# In[5]:


class MCDropout(keras.layers.Dropout) :
    def call(self, inputs) :
        return suerp().call(inputs, training = True)
    
class HourGlassLayerUnit(keras.layers.Layer) :
    def __init__(self, filters, pooling = True, strides = 1, dropout = None, activation = "relu", **kwargs) :
        super().__init__(**kwargs)
        
        self.dropout_1 = None
        self.dropout_2 = None
        self.dropout_3 = None

        if dropout is not None :
            self.dropout_1 = MCDropout(rate = dropout)
            self.dropout_2 = MCDropout(rate = dropout)
            self.dropout_3 = MCDropout(rate = dropout)
        
        self.conv_1 = keras.layers.Conv2D(filters = filters // 2, strides = strides, kernel_size = (1, 1), activation = None, padding = "same")
        self.bn_1 = keras.layers.BatchNormalization()
        self.act_1 = keras.layers.Activation(activation)

        self.conv_2 = keras.layers.Conv2D(filters = filters // 2, strides = strides, kernel_size = (3, 3), activation = None, padding = "same")
        self.bn_2 = keras.layers.BatchNormalization()
        self.act_2 = keras.layers.Activation(activation)

        self.conv_3 = keras.layers.Conv2D(filters = filters, strides = strides, kernel_size = (1, 1), activation = None, padding = "same")
        self.bn_3 = keras.layers.BatchNormalization()
        self.act_3 = keras.layers.Activation(activation)
        
        self.pool = None
        if pooling is True :
            self.pool = keras.layers.MaxPooling2D()
        
    def call(self, x) :
        inputs = x
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.act_1(x)
        if self.dropout_1 is not None :
            x = self.dropout_1(x)
        
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.act_2(x)
        
        if self.dropout_2 is not None :
            x = self.dropout_2(x)
            
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.act_3(x)
        
        if self.dropout_3 is not None :
            x = self.dropout_3(x)
        
        x = keras.layers.Add()([inputs, x])
        
        if self.pool is not None :
            x = self.pool(x)
        return x
    
class HourGlassSkipUnit(keras.layers.Layer) :
    def __init__(self, filters, activation = "relu", dropout = None, **kwargs) :
        self.conv = keras.layers.Conv2D(filters = filters, kernel_size = 1, strides = 1, padding = "same", activation = None)
        self.bn = keras.layers.BatchNormalization()
        self.act = keras.layers.Activation(activation)
        self.dropout = None
        if dropout is not None :
            self.dropout = MCDropout(rate = dropout)
            
        super().__init__(**kwargs)
        
    def call(self, x) :
        inputs = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        if self.dropout is not None :
            x = self.dropout(x)
        return keras.layers.Add()([inputs, x])
    
class HourGlassPreprocessLayer(keras.layers.Layer) :
    def __init__(self, filters, activation = "relu", **kwargs) :
        self.conv = keras.layers.Conv2D(filters = filters, kernel_size = (7, 7), strides = 2, padding = "same", activation = None)
        self.bn = keras.layers.BatchNormalization()
        self.res = HourGlassLayerUnit(filters = filters)
        self.act = keras.layers.Activation(activation)
        self.pooling = keras.layers.MaxPooling2D()
        super().__init__(**kwargs)
        
    def call(self, x) :
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.res(x)
#        x = self.pooling(x)
        return x
    
class HourGlassOutputLayer(keras.layers.Layer) :
    def __init__(self, activation = "relu", **kwargs) :
        self.heatmap_size_matcher = keras.layers.UpSampling2D(size = 4)
        self.heatmap_output = keras.layers.Conv2D(kernel_size = 1, filters = 24, padding = "same", activation = None)
        self.heatmap_bn = keras.layers.BatchNormalization()
        self.heatmap_act = keras.layers.Activation("sigmoid", name = "heatmap_output")
        
        self.center = keras.layers.Dense(4, name = "center_output")
        
        super().__init__(**kwargs)
        
    def call(self, x) :
        heatmap = self.heatmap_size_matcher(x)
        heatmap = self.heatmap_output(heatmap)
        heatmap = self.heatmap_bn(heatmap)
        heatmap = self.heatmap_act(heatmap)
        
        center = tf.reduce_sum(heatmap, axis = -1, keepdims = False)
        center = keras.layers.Flatten()(center)
        center = self.center(center)
        
        return [heatmap, center]


# In[10]:


class HourGlassModule(keras.layers.Layer) :
    def __init__(self, filters, activation = "relu", strides = 1, dropout = None, padding = "same", **kwargs) :
        
        self.enc_1 = HourGlassLayerUnit(filters, activation = activation)
        self.enc_2 = HourGlassLayerUnit(filters, activation = activation)
        self.enc_3 = HourGlassLayerUnit(filters, activation = activation)
        self.enc_4 = HourGlassLayerUnit(filters, activation = activation)
        
        self.skip_1 = HourGlassSkipUnit(filters, activation = activation)
        self.skip_2 = HourGlassSkipUnit(filters, activation = activation)
        self.skip_3 = HourGlassSkipUnit(filters, activation = activation)
        self.skip_4 = HourGlassSkipUnit(filters, activation = activation)
        
        self.middle_1 = HourGlassLayerUnit(filters = filters, pooling = False, activation = activation)
        self.middle_2 = HourGlassLayerUnit(filters = filters, pooling = False, activation = activation)
        self.middle_3 = HourGlassLayerUnit(filters = filters, pooling = False, activation = activation)
        
        self.dec_1 = keras.layers.UpSampling2D()
        self.dec_2 = keras.layers.UpSampling2D()
        self.dec_3 = keras.layers.UpSampling2D()
        self.dec_4 = keras.layers.UpSampling2D()
        
        self.conv1 = HourGlassLayerUnit(filters = filters, pooling = False, activation = activation)
        self.conv2 = HourGlassLayerUnit(filters = filters, pooling = False, activation = activation)
        self.conv3 = HourGlassLayerUnit(filters = filters, pooling = False, activation = activation)
        
        self.semi_output = HourGlassOutputLayer(activation = activation)
        self.heatmap_size_matcher = keras.layers.MaxPooling2D(pool_size = 4)
        self.channel_matcher = keras.layers.Conv2D(filters = filters, kernel_size = 1, padding = "valid", activation = None, strides = 1)
        
        super().__init__(**kwargs)
        
        
    def call(self, x) :
        inputs = x
        enc_1 = self.enc_1(x)
        enc_2 = self.enc_2(enc_1)
        enc_3 = self.enc_3(enc_2)
        enc_4 = self.enc_4(enc_3)
        
        res_1 = self.skip_1(enc_1)
        res_2 = self.skip_2(enc_2)
        res_3 = self.skip_3(enc_3)
        res_4 = self.skip_3(enc_4)
        
        middle = self.middle_1(enc_4)
        middle = self.middle_2(middle)
        middle = self.middle_3(middle)
        
        add = keras.layers.Add()([middle, res_4])
        up = self.dec_1(add)
        
        add = keras.layers.Add()([up, res_3])
        up = self.dec_2(add)
        
        add = keras.layers.Add()([up, res_2])
        up = self.dec_3(add)
        
        add = keras.layers.Add()([up, res_1])
        up = self.dec_4(add)
        
        conv1 = self.conv1(up)
        conv2 = self.conv2(conv1)
        
        output_heatmap, center = self.semi_output(conv1)
        heatmap = self.channel_matcher(output_heatmap)
        heatmap = self.heatmap_size_matcher(heatmap)
        conv3 = self.conv3(heatmap)

        add = keras.layers.Add()([inputs, conv2, conv3])
        
        return [add, output_heatmap, center]


# In[ ]:




