import keras
from keras.layers import Input, Conv2D, Lambda, Flatten, Dense, Concatenate
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input image into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Apply different convolutional kernels to each group
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_layer[0])
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(split_layer[1])
    conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(split_layer[2])
    
    # Concatenate the outputs from these groups
    concatenated = Concatenate()([conv1x1, conv3x3, conv5x5])
    
    # Flatten the concatenated output
    flattened = Flatten()(concatenated)
    
    # Pass the flattened output through two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model