import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, UpSampling2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First stage of convolution and max pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    
    # Second stage of convolution and max pooling
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    # Additional convolutional and dropout layers
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    dropout1 = Dropout(0.5)(conv3)
    
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout1)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)
    dropout2 = Dropout(0.5)(conv4)
    
    # Upsampling with skip connections
    upsample1 = UpSampling2D(size=(2, 2))(dropout2)
    concat1 = Concatenate()([conv2, upsample1])
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat1)
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
    
    upsample2 = UpSampling2D(size=(2, 2))(conv5)
    concat2 = Concatenate()([conv1, upsample2])
    conv6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat2)
    conv6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv6)
    
    # 1x1 convolutional layer to produce probability outputs
    conv7 = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(conv6)
    
    # Flatten and dense layers
    flatten_layer = Flatten()(conv7)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model