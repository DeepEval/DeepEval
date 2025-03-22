import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    
    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm1)
    batch_norm2 = BatchNormalization()(conv2)
    
    # Third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm2)
    batch_norm3 = BatchNormalization()(conv3)
    
    # Average pooling layers
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=1)(batch_norm3)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=2)(batch_norm3)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=4)(batch_norm3)
    
    # Concatenate the outputs of the pooling layers
    concatenated = Concatenate()([avg_pool1, avg_pool2, avg_pool3])
    
    # Flatten the concatenated output
    flattened = Flatten()(concatenated)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flattened)
    dense2 = Dense(units=128, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model