import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Concatenate, UpSampling2D, ZeroPadding2D, Conv2DTranspose, BatchNormalization, Activation, AveragePooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    relu1 = Activation('relu')(batch_norm1)
    
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(relu1)
    batch_norm2 = BatchNormalization()(conv2)
    relu2 = Activation('relu')(batch_norm2)
    
    maxpool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(relu2)
    
    # Branch path
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)
    batch_norm3 = BatchNormalization()(conv3)
    relu3 = Activation('relu')(batch_norm3)
    
    maxpool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(relu3)
    
    up_conv1 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=2, padding='same')(maxpool2)
    batch_norm4 = BatchNormalization()(up_conv1)
    relu4 = Activation('relu')(batch_norm4)
    
    concat = Concatenate(axis=-1)([relu4, conv2])
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(concat)
    batch_norm5 = BatchNormalization()(conv4)
    relu5 = Activation('relu')(batch_norm5)
    
    up_conv2 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=2, padding='same')(relu5)
    batch_norm6 = BatchNormalization()(up_conv2)
    relu6 = Activation('relu')(batch_norm6)
    
    concat = Concatenate(axis=-1)[[relu6, conv1]]
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(concat)
    batch_norm7 = BatchNormalization()(conv5)
    relu7 = Activation('relu')(batch_norm7)
    
    output = Conv2D(filters=3, kernel_size=(1, 1), padding='same')(relu7)
    output = Activation('softmax')(output)
    
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model