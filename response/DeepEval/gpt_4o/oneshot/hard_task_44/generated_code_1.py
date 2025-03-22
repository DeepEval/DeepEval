import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Dense, Lambda

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        # Split the input tensor into 3 groups along the channel axis
        split1, split2, split3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Feature extraction on each split
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split1)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split2)
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split3)
        
        # Apply dropout to reduce overfitting
        dropout1 = Dropout(rate=0.3)(conv1)
        dropout2 = Dropout(rate=0.3)(conv2)
        dropout3 = Dropout(rate=0.3)(conv3)
        
        # Concatenate the outputs
        concat = Concatenate()([dropout1, dropout2, dropout3])
        
        return concat
    
    block1_output = block1(input_layer)

    # Block 2
    def block2(input_tensor):
        # Branch 1: 1x1 Convolution
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        # Branch 2: 1x1 Convolution followed by 3x3 Convolution
        branch2_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2_conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_conv1)

        # Branch 3: 1x1 Convolution followed by 5x5 Convolution
        branch3_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3_conv2 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch3_conv1)

        # Branch 4: 3x3 Max Pooling followed by 1x1 Convolution
        branch4_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch4_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4_pool)

        # Concatenate all the branches
        concat = Concatenate()([branch1, branch2_conv2, branch3_conv2, branch4_conv])
        
        return concat

    block2_output = block2(block1_output)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(block2_output)
    fc_layer = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(fc_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model