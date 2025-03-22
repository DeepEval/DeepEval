import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, Flatten, Dense, Lambda

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block_1(input_tensor):
        # Split input into three groups along the channel axis
        inputs_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply convolution with different kernel sizes
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(inputs_groups[2])
        
        # Apply dropout to reduce overfitting
        conv1 = Dropout(rate=0.2)(conv1)
        conv2 = Dropout(rate=0.2)(conv2)
        conv3 = Dropout(rate=0.2)(conv3)
        
        # Concatenate the outputs
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    # Block 2
    def block_2(input_tensor):
        # Branch 1: 1x1 convolution
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Branch 2: 1x1 convolution followed by 3x3 convolution
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        
        # Branch 3: 1x1 convolution followed by 5x5 convolution
        branch3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(branch3)
        
        # Branch 4: 3x3 max pooling followed by 1x1 convolution
        branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(branch4)
        
        # Concatenate all branches
        output_tensor = Concatenate()([branch1, branch2, branch3, branch4])
        return output_tensor

    # Constructing the model
    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)
    
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model