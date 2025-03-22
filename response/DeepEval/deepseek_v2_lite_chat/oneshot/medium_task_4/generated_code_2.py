import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: Two blocks of convolution and average pooling
    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
        return pool2

    path1_output = block(input_tensor=input_layer)
    
    # Path 2: Single convolutional layer
    def block(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        avg_pool = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(conv1)
        return avg_pool

    path2_output = block(input_tensor=input_layer)
    
    # Combine Path 1 and Path 2
    combined_output = Concatenate(axis=-1)([path1_output, path2_output])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(combined_output)
    flatten = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()