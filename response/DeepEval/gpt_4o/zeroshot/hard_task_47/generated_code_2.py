import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, SeparableConv2D, BatchNormalization, Concatenate, Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First Block
    # Split the input into 3 along the last dimension
    split_input = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Depthwise separable convolutions with different kernel sizes
    conv_1x1 = SeparableConv2D(32, (1, 1), padding='same', activation='relu')(split_input[0])
    conv_1x1 = BatchNormalization()(conv_1x1)

    conv_3x3 = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(split_input[1])
    conv_3x3 = BatchNormalization()(conv_3x3)

    conv_5x5 = SeparableConv2D(32, (5, 5), padding='same', activation='relu')(split_input[2])
    conv_5x5 = BatchNormalization()(conv_5x5)

    # Concatenate outputs of the first block
    concatenated_1st_block = Concatenate()([conv_1x1, conv_3x3, conv_5x5])

    # Second Block
    # Branch 1: <1x1 convolution, 3x3 convolution>
    branch_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(concatenated_1st_block)
    branch_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch_1)

    # Branch 2: <1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution>
    branch_2 = Conv2D(32, (1, 1), padding='same', activation='relu')(concatenated_1st_block)
    branch_2 = Conv2D(32, (1, 7), padding='same', activation='relu')(branch_2)
    branch_2 = Conv2D(32, (7, 1), padding='same', activation='relu')(branch_2)
    branch_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch_2)

    # Branch 3: Average pooling
    branch_3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(concatenated_1st_block)

    # Concatenate outputs of the second block
    concatenated_2nd_block = Concatenate()([branch_1, branch_2, branch_3])
    
    # Fully connected layers
    flat = Flatten()(concatenated_2nd_block)
    fc1 = Dense(128, activation='relu')(flat)
    output_layer = Dense(10, activation='softmax')(fc1)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Usage
model = dl_model()
model.summary()