import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Concatenate, Dense, Reshape, Lambda, SeparableConv2D
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First Block: Multiple MaxPooling with different scales
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
    
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)
    
    # Apply Dropout
    drop1 = Dropout(0.3)(flat1)
    drop2 = Dropout(0.3)(flat2)
    drop3 = Dropout(0.3)(flat3)
    
    # Concatenate the flattened outputs
    concat1 = Concatenate()([drop1, drop2, drop3])
    
    # Fully connected layer and reshape
    dense1 = Dense(units=1024, activation='relu')(concat1)
    reshape1 = Reshape((8, 8, 16))(dense1)  # Reshaping to 4D tensor for next block
    
    # Second Block: Splitting and Separable Convolutional processing
    def split_and_process(x):
        # Split into 4 groups
        split = tf.split(x, num_or_size_splits=4, axis=-1)
        # Separable convolution with different kernel sizes
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split[0])
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split[1])
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split[2])
        conv4 = SeparableConv2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu')(split[3])
        # Concatenate the outputs
        return Concatenate()([conv1, conv2, conv3, conv4])
    
    processed = Lambda(split_and_process)(reshape1)
    
    # Final Dense Layer for classification
    flat2 = Flatten()(processed)
    output_layer = Dense(units=10, activation='softmax')(flat2)
    
    # Constructing the Model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model