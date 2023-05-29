# -*- coding: utf-8 -*-

#### MODEL ###
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization,concatenate,Conv2DTranspose,Dropout
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers import Activation, BatchNormalization, add, Reshape
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras import backend as K


def relu6(x):
    return K.relu(x, max_value=6.0)


def _conv_block(inputs, filters, kernel, strides):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu6)(x)


def _bottleneck(inputs, filters, kernel, t, s, r=False):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t
    print(tchannel)

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = add([x, inputs])
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    x = _bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)

    return x


# def MobileNetv2(input_shape, k):
#     inputs = Input(shape=input_shape)
#     x = _conv_block(inputs, 32, (3, 3), strides=(2, 2))
#
#     x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)
#     x = _inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)
#     x = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)
#     x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=4)
#     x = _inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=3)
#     x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
#     x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)
#
#     # x = _conv_block(x, 1280, (1, 1), strides=(1, 1))
#     # x = GlobalAveragePooling2D()(x)
#     # x = Reshape((1, 1, 1280))(x)
#     # x = Dropout(0.3, name='Dropout')(x)
#     # x = Conv2D(k, (1, 1), padding='same')(x)
#     #
#     # x = Activation('softmax', name='softmax')(x)
#     output = Reshape((k,))(x)
#
#     model = Model(inputs, output)
#
#     return model







def Mobilev2_UNET (input_shape=(512,512,1),last_activation='sigmoid'):
    inputs=Input(shape=input_shape)

    
    # conv1 = Conv2D(32,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    # d1=Dropout(0.1)(conv1)
    # conv2 = Conv2D(32,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d1)
    # b=BatchNormalization()(conv2)
    #
    # pool1 = MaxPooling2D(pool_size=(2, 2))(b)
    # conv3 = Conv2D(64,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    # d2=Dropout(0.2)(conv3)
    # conv4 = Conv2D(64,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d2)
    # b1=BatchNormalization()(conv4)
    #
    # pool2 = MaxPooling2D(pool_size=(2, 2))(b1)
    # conv5 = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    # d3=Dropout(0.3)(conv5)
    # conv6 = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d3)
    # b2=BatchNormalization()(conv6)
    #
    # pool3 = MaxPooling2D(pool_size=(2, 2))(b2)
    # conv7 = Conv2D(256,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    # d4=Dropout(0.4)(conv7)
    # conv8 = Conv2D(256,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d4)
    # b3=BatchNormalization()(conv8)
    #
    # pool4 = MaxPooling2D(pool_size=(2, 2))(b3)
    # conv9 = Conv2D(512,(3,3),activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    # d5=Dropout(0.5)(conv9)
    # conv10 = Conv2D(512,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d5)
    # b4=BatchNormalization()(conv10)
    conv1 = _conv_block(inputs, 32, (3, 3), strides=(1, 1))
    conv1 = Dropout(0.1)(conv1)

    conv2 = _inverted_residual_block(conv1, 32, (3, 3), t=1, strides=1, n=1)
    conv2 = Dropout(0.2)(conv2)
    conv3 = _inverted_residual_block(conv2, 64, (3, 3), t=1, strides=2, n=1)
    conv3 = Dropout(0.3)(conv3)
    conv4 = _inverted_residual_block(conv3, 128, (3, 3), t=1, strides=2, n=2)
    conv4 = Dropout(0.4)(conv4)
    conv5 = _inverted_residual_block(conv4, 256, (3, 3), t=1, strides=2, n=2)
    conv5 = Dropout(0.5)(conv5)
    conv6 = _inverted_residual_block(conv5, 512, (3, 3), t=1, strides=2, n=1)
    # x = _inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=3)
    # x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
    # x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)
    # print(conv1)
    # print(conv2)
    # print(conv3)
    # print(conv4)
    print(conv5)
    # print(conv6)

    conv11 = Conv2DTranspose(512,(4,4), activation = 'relu', padding = 'same', strides=(2,2),kernel_initializer = 'he_normal')(conv6)
    print(conv11)
    x= concatenate([conv11,conv5])
    conv12 = Conv2D(256,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    d6=Dropout(0.4)(conv12)
    conv13 = Conv2D(256,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d6)
    b5=BatchNormalization()(conv13)
    
    
    conv14 = Conv2DTranspose(256,(4,4), activation = 'relu', padding = 'same', strides=(2,2),kernel_initializer = 'he_normal')(b5)
    x1=concatenate([conv14,conv4])
    conv15 = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x1)
    d7=Dropout(0.3)(conv15)
    conv16 = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d7)
    b6=BatchNormalization()(conv16)
    
    conv17 = Conv2DTranspose(128,(4,4), activation = 'relu', padding = 'same',strides=(2,2), kernel_initializer = 'he_normal')(b6)
    x2=concatenate([conv17,conv3])
    conv18 = Conv2D(64,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x2)
    d8=Dropout(0.2)(conv18)
    conv19 = Conv2D(64,(3,3) ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d8)
    b7=BatchNormalization()(conv19)
    
    conv20 = Conv2DTranspose(64,(4,4), activation = 'relu', padding = 'same',strides=(2,2), kernel_initializer = 'he_normal')(b7)
    x3=concatenate([conv20,conv2])
    conv21 = Conv2D(32,(3,3) ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x3)
    d9=Dropout(0.1)(conv21)
    conv22 = Conv2D(32,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d9)
    # print("---------------------------------------------------")
    # print(conv11)
    # print(conv12)
    # print(conv13)
    # print(conv14)
    # print(conv15)
    # print(conv16)
    # print(conv17)
    # print(conv18)
    # print(conv19)
    # print(conv20)
    # print(conv21)
    # print(conv22)
    outputs = Conv2D(1,(1,1), activation = last_activation, padding = 'same', kernel_initializer = 'he_normal')(conv22)
    model2 = Model( inputs = inputs, outputs = outputs)
    
    return model2

def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta,
                                cmd='op',
                                options=opts)
    return flops.total_float_ops


if __name__ == '__main__':
    model = Mobilev2_UNET(input_shape=(512,512,1),last_activation='sigmoid')
    model.summary()
    # print(get_flops(model))






