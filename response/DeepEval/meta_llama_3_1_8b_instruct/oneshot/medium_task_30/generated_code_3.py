import keras
from keras.layers import Input, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    
    # Define the input layer with shape (32, 32, 3) for CIFAR-10 dataset
    input_layer = Input(shape=(32, 32, 3))
    
    # Apply three average pooling layers with different window sizes
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1)(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2)(pool1)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=4)(pool2)
    
    # Flatten the outputs of the pooling layers and concatenate them
    flattened_pool1 = Flatten()(pool1)
    flattened_pool2 = Flatten()(pool2)
    flattened_pool3 = Flatten()(pool3)
    concat = Concatenate()([flattened_pool1, flattened_pool2, flattened_pool3])
    
    # Further flatten the concatenated features and process through two fully connected layers
    flatten_layer = Flatten()(concat)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Create the Keras model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model