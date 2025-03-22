import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, Concatenate, Flatten, Dense, Lambda
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Apply convolution with varying kernel sizes
    conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_channels[0])
    conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_channels[1])
    conv_5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_channels[2])

    # Apply Dropout
    dropout_layer = Dropout(0.5)(Concatenate()([conv_1x1, conv_3x3, conv_5x5]))

    # Block 2
    def block_2(input_tensor):
        # Branch 1: 1x1 convolution
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Branch 2: 1x1 convolution followed by 3x3 convolution
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        
        # Branch 3: 1x1 convolution followed by 5x5 convolution
        branch3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(branch3)
        
        # Branch 4: 3x3 max pooling followed by 1x1 convolution
        branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch4 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(branch4)

        return Concatenate()([branch1, branch2, branch3, branch4])
    
    block_2_output = block_2(dropout_layer)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(block_2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model