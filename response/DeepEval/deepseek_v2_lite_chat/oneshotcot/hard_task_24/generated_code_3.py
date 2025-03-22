import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional Layer
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 1: 3x3 Convolution
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Branch 2: Max Pooling -> 3x3 Convolution -> UpSampling -> Restore Size
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool)
    upsample = UpSampling2D()(conv3)
    
    # Branch 3: Max Pooling -> 3x3 Convolution -> UpSampling -> Restore Size
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool)
    upsample = UpSampling2D()(conv4)
    
    # Concatenate all branches
    fuse = Concatenate()([conv2, conv3, conv4])

    # Additional 1x1 Convolution
    conv5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(fuse)
    
    # Batch Normalization
    bn = BatchNormalization()(conv5)
    
    # Flatten
    flatten = Flatten()(bn)
    
    # Dense Layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()
model.summary()