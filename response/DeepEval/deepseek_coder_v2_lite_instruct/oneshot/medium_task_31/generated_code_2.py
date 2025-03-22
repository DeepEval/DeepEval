import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Dropout
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Define the convolutional blocks for each group
    def conv_block(input_tensor, filters, kernel_size):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        return batch_norm
    
    # Apply different convolutional kernels to each group
    conv1x1 = conv_block(split[0], filters=32, kernel_size=(1, 1))
    conv3x3 = conv_block(split[1], filters=32, kernel_size=(3, 3))
    conv5x5 = conv_block(split[2], filters=32, kernel_size=(5, 5))
    
    # Concatenate the outputs of the three groups
    concatenated = Concatenate()([conv1x1, conv3x3, conv5x5])
    
    # Flatten the concatenated output
    flatten = Flatten()(concatenated)
    
    # Pass through two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(units=64, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)
    output_layer = Dense(units=10, activation='softmax')(dropout2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model