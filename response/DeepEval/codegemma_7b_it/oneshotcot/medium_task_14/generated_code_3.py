import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    relu1 = Activation('relu')(batch_norm1)
    
    # Block 2
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu1)
    batch_norm2 = BatchNormalization()(conv2)
    relu2 = Activation('relu')(batch_norm2)
    
    # Block 3
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu2)
    batch_norm3 = BatchNormalization()(conv3)
    relu3 = Activation('relu')(batch_norm3)
    
    # Parallel branch
    conv4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm4 = BatchNormalization()(conv4)
    relu4 = Activation('relu')(batch_norm4)
    
    # Concatenate paths
    concat = Add()([relu3, relu4])
    
    # Fully connected layers
    flatten = keras.layers.Flatten()(concat)
    dense1 = Dense(units=64, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model