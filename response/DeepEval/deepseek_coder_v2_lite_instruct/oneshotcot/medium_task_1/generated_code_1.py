import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, BatchNormalization
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    
    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm1)
    batch_norm2 = BatchNormalization()(conv2)
    
    # Max-pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(batch_norm2)
    
    # Add the max-pooling output to the original input
    added = Add()([max_pooling, input_layer])
    
    # Flatten the output
    flattened = Flatten()(added)
    
    # First fully connected layer
    dense1 = Dense(units=256, activation='relu')(flattened)
    batch_norm_fc1 = BatchNormalization()(dense1)
    
    # Second fully connected layer
    dense2 = Dense(units=128, activation='relu')(batch_norm_fc1)
    batch_norm_fc2 = BatchNormalization()(dense2)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(batch_norm_fc2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model