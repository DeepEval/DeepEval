import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Concatenate, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 1x1 convolution
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1 = Dropout(0.2)(conv1)
    
    # Branch 2: 1x1 convolution followed by 3x3 convolution
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Dropout(0.2)(conv2)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    conv3 = Dropout(0.2)(conv3)
    
    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv4 = Dropout(0.2)(conv4)
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)
    conv5 = Dropout(0.2)(conv5)
    conv6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
    conv6 = Dropout(0.2)(conv6)
    
    # Branch 4: Average pooling followed by 1x1 convolution
    pool = AveragePooling2D(pool_size=(8, 8), strides=(8, 8), padding='same')(input_layer)
    conv7 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool)
    conv7 = Dropout(0.2)(conv7)
    
    # Concatenate the outputs from all branches
    output_tensor = Concatenate()([conv1, conv3, conv6, conv7])
    
    # Flatten the output
    flatten_layer = Flatten()(output_tensor)
    
    # Three fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense1 = Dropout(0.2)(dense1)
    dense2 = Dense(units=64, activation='relu')(dense1)
    dense2 = Dropout(0.2)(dense2)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model