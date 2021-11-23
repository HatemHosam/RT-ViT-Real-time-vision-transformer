import tensorflow as tf
from tensorflow.keras.models import Model
import os
import numpy as np
from tensorflow.keras.layers import Input, MaxPooling2D, Activation, Add, Conv2DTranspose, SeparableConv2D, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, UpSampling2D
import tensorflow_addons as tfa
from vit_keras import vit




# encoder architectures
#base ViT-b16


def ViT_b16_US_decoder (dataset= 'NYUV2'):
    #encoder
    if dataset= ='NYUV2':
        IMAGE_SIZE = (448,448)
    elif dataset= ='CS':
        IMAGE_SIZE = (320,640) 
    else:
        print('The available image sizes are (320,640) for CS or (448,448) for NYUV2 datasets')
    
    vit_model = vit.vit_b16(
            image_size = IMAGE_SIZE,
            activation = 'softmax',
            pretrained = True,
            include_top = False,
            pretrained_top = False)
            
    x = tf.keras.layers.Lambda(lambda v: v[:,1:,:])(vit_model.layers[-2].output)
    if dataset= ='NYUV2':
        x = tf.keras.layers.Reshape((28,28,768))(x)
    elif dataset= ='CS':
        x = tf.keras.layers.Reshape((20,40,768))(x)
    else:
        print('The available weights are for CS or NYUV2 datasets')
    
    #decoder
    x = UpSampling2D(size = (2,2))(x)
    x = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)

    x = UpSampling2D(size = (2,2))(x)
    x = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)

    x = UpSampling2D(size = (2,2))(x)
    x = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)

    x = UpSampling2D(size = (2,2))(x)
    x = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    out = Conv2D(1, 1, activation = 'linear')(x)
    model = tf.keras.models.Model(vit_model.input, out)
     #load pre-trained weights
     if dataset= ='NYUV2':
        model.load_weights('weights/ViT_b16_US_decoder_NYU.h5')
    elif dataset= ='CS':
        model.load_weights('weights/ViT_b16_US_decoder_CS.h5')
    else:
        print('The available weights are for CS or NYUV2 datasets')
    return model
    
    
def ViT_s16_US_decoder2 (dataset= 'NYUV2'):
    #encoder
    if dataset= ='NYUV2':
        IMAGE_SIZE = (448,448)
    elif dataset= ='CS':
        IMAGE_SIZE = (320,640) 
    else:
        print('The available image sizes are (320,640) for CS or (448,448) for NYUV2 datasets')
    
    vit_model = vit.vit_b16(
            image_size = IMAGE_SIZE,
            activation = 'softmax',
            pretrained = True,
            include_top = False,
            pretrained_top = False)
            
    x,_ = vit_model.get_layer('Transformer/encoderblock_5').output
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="Transformer/encoder_norm")(x)
    x = tf.keras.layers.Lambda(lambda v: v[:,1:,:])(x)
    if dataset= ='NYUV2':
        x = tf.keras.layers.Reshape((28,28,768))(x)
    elif dataset= ='CS':
        x = tf.keras.layers.Reshape((20,40,768))(x)
    else:
        print('The available weights are for CS or NYUV2 datasets')
    
    #decoder
    x = UpSampling2D(size = (2,2))(x)
    x = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)

    x = UpSampling2D(size = (2,2))(x)
    x = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)

    x = UpSampling2D(size = (2,2))(x)
    x = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)

    x = UpSampling2D(size = (2,2))(x)
    x = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    out = Conv2D(1, 1, activation = 'linear')(x)
    model = tf.keras.models.Model(vit_model.input, out)
    #load pre-trained weights
     if dataset= ='NYUV2':
        model.load_weights('weights/ViT_s16_US_deconv2_NYU.h5')
    elif dataset= ='CS':
        model.load_weights('weights/ViT_s16_US_deconv2_CS.h5')
    else:
        print('The available weights are for CS or NYUV2 datasets')
    return model

def ViT_s16_deconv_decoder (dataset= 'NYUV2'):
    #encoder
    if dataset= ='NYUV2':
        IMAGE_SIZE = (448,448)
    elif dataset= ='CS':
        IMAGE_SIZE = (320,640) 
    else:
        print('The available image sizes are (320,640) for CS or (448,448) for NYUV2 datasets')
    
    vit_model = vit.vit_b16(
            image_size = IMAGE_SIZE,
            activation = 'softmax',
            pretrained = True,
            include_top = False,
            pretrained_top = False)
            
    x,_ = vit_model.get_layer('Transformer/encoderblock_5').output
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="Transformer/encoder_norm")(x)
    x = tf.keras.layers.Lambda(lambda v: v[:,1:,:])(x)
    if dataset= ='NYUV2':
        x = tf.keras.layers.Reshape((28,28,768))(x)
    elif dataset= ='CS':
        x = tf.keras.layers.Reshape((20,40,768))(x)
    else:
        print('The available weights are for CS or NYUV2 datasets')
    
     #decoder
    # UP 1
    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    # UP 2
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    # UP 3
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    # UP 4
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    out = Conv2D(1, (1, 1), padding='same')(x)
    model = tf.keras.models.Model(vit_model.input, out)
    #load pre-trained weights
     if dataset= ='NYUV2':
        model.load_weights('weights/ViT_s16_deconv_decoder_NYU.h5')
    elif dataset= ='CS':
        model.load_weights('weights/ViT_s16_deconv_decoder_CS.h5')
    else:
        print('The available weights are for CS or NYUV2 datasets')
    return model

def ViT_t16_US_decoder2 (dataset= 'NYUV2'):
    #encoder
    if dataset= ='NYUV2':
        IMAGE_SIZE = (448,448)
    elif dataset= ='CS':
        IMAGE_SIZE = (320,640) 
    else:
        print('The available image sizes are (320,640) for CS or (448,448) for NYUV2 datasets')
    
    vit_model = vit.vit_b16(
            image_size = IMAGE_SIZE,
            activation = 'softmax',
            pretrained = True,
            include_top = False,
            pretrained_top = False)
            
    x,_ = vit_model.get_layer('Transformer/encoderblock_3').output
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="Transformer/encoder_norm")(x)
    x = tf.keras.layers.Lambda(lambda v: v[:,1:,:])(x)
    if dataset= ='NYUV2':
        x = tf.keras.layers.Reshape((28,28,768))(x)
    elif dataset= ='CS':
        x = tf.keras.layers.Reshape((20,40,768))(x)
    else:
        print('The available weights are for CS or NYUV2 datasets')
        
    #Upsampling decoder 2
    x = UpSampling2D(size = (2,2))(x)
    x = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)

    x = UpSampling2D(size = (2,2))(x)
    x = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)

    x = UpSampling2D(size = (2,2))(x)
    x = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)

    x = UpSampling2D(size = (2,2))(x)
    x = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    out = Conv2D(1, 1, activation = 'linear')(x)
    model = tf.keras.models.Model(vit_model.input, out)
    #load pre-trained weights
     if dataset= ='NYUV2':
        model.load_weights('weights/ViT_t16_US_decoder2_NYU.h5')
    elif dataset= ='CS':
        model.load_weights('weights/ViT_t16_US_decoder2_CS.h5')
    else:
        print('The available weights are for CS or NYUV2 datasets')
    return model

def ViT_t16_deconv_decoder (dataset= 'NYUV2'):
    #encoder
    if dataset= ='NYUV2':
        IMAGE_SIZE = (448,448)
    elif dataset= ='CS':
        IMAGE_SIZE = (320,640) 
    else:
        print('The available image sizes are (320,640) for CS or (448,448) for NYUV2 datasets')
    
    vit_model = vit.vit_b16(
            image_size = IMAGE_SIZE,
            activation = 'softmax',
            pretrained = True,
            include_top = False,
            pretrained_top = False)
            
    x,_ = vit_model.get_layer('Transformer/encoderblock_3').output
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="Transformer/encoder_norm")(x)
    x = tf.keras.layers.Lambda(lambda v: v[:,1:,:])(x)
    if dataset= ='NYUV2':
        x = tf.keras.layers.Reshape((28,28,768))(x)
    elif dataset= ='CS':
        x = tf.keras.layers.Reshape((20,40,768))(x)
    else:
        print('The available weights are for Cityscapes or NYUV2 datasets')
     
    #decoder
    # UP 1
    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    # UP 2
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    # UP 3
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    # UP 4
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    out = Conv2D(1, (1, 1), padding='same')(x)
    model = tf.keras.models.Model(vit_model.input, out)
    #load pre-trained weights
     if dataset= ='NYUV2':
        model.load_weights('weights/ViT_t16_deconv_decoder_NYU.h5')
    elif dataset= ='CS':
        model.load_weights('weights/ViT_t16_deconv_decoder_CS.h5')
    else:
        print('The available weights are for CS or NYUV2 datasets')
    return model
    
def ViT_t16_DS_decoder (dataset= 'NYUV2'):
    #encoder
    if dataset= ='NYUV2':
        IMAGE_SIZE = (448,448)
    elif dataset= ='CS':
        IMAGE_SIZE = (320,640) 
    else:
        print('The available image sizes are (320,640) for CS or (448,448) for NYUV2 datasets')
    
    vit_model = vit.vit_b16(
            image_size = IMAGE_SIZE,
            activation = 'softmax',
            pretrained = True,
            include_top = False,
            pretrained_top = False)
            
    x,_ = vit_model.get_layer('Transformer/encoderblock_3').output
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="Transformer/encoder_norm")(x)
    x = tf.keras.layers.Lambda(lambda v: v[:,1:,:])(x)
    if dataset= ='NYUV2':
        x = tf.keras.layers.Reshape((28,28,768))(x)
    elif dataset= ='CS':
        x = tf.keras.layers.Reshape((20,40,768))(x)
    else:
        print('The available weights are for CS or NYUV2 datasets')
     
    #decoder
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)

    out = tf.nn.depth_to_space(x, 16)
    model = tf.keras.models.Model(vit_model.input, out)
    #load pre-trained weights
     if dataset= ='NYUV2':
        model.load_weights('weights/ViT_t16_DS_decoder_NYU.h5')
    elif dataset= ='CS':
        model.load_weights('weights/ViT_t16_DS_decoder_CS.h5')
    else:
        print('The available weights are for CS or NYUV2 datasets')
    return model
