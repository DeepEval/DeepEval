import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    # Max-pooling layer
    pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    # Add the pooled features with the input layer
    add = Add()([pool, input_layer])
    
    # Batch normalization
    batch_norm = BatchNormalization()(add)
    
    # Flatten the output
    flatten = Flatten()(batch_norm)
    
    # Two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model