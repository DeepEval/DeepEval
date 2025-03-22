import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial 1x1 convolutional layer
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 1: 3x3 convolutional layer
    branch1_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Branch 2 and 3: Sequential layers with max pooling, 3x3 conv, and upsampling
    branch2_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2_pool = MaxPooling2D(pool_size=(2, 2))(branch2_conv)
    branch2_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_pool)
    branch2_upsample = UpSampling2D()(branch2_conv2)
    
    branch3_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3_pool = MaxPooling2D(pool_size=(2, 2))(branch3_conv)
    branch3_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3_pool)
    branch3_upsample = UpSampling2D()(branch3_conv2)
    
    # Concatenate the outputs of the branches
    concat_layer = Concatenate(axis=-1)([branch1_conv, branch2_conv2, branch3_upsample])
    
    # Additional 1x1 convolutional layer
    conv_layer = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_layer)
    
    # Flatten and fully connected layers for classification
    flatten = Flatten()(conv_layer)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and return the model
model = dl_model()
model.summary()