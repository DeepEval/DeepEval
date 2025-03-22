import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, GlobalAveragePooling2D, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Split input into 3 groups and apply separable convolutions
    def block_1(input_tensor):
        # Splitting the input into 3 groups
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Applying depthwise separable convolutions with different kernel sizes
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(inputs_groups[2])
        
        # Concatenating the outputs from all three convolutions
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    # Block 2: Multiple branches for feature extraction
    def block_2(input_tensor):
        # 3x3 convolution branch
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        
        # 1x1 convolution followed by two 3x3 convolutions branch
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)

        # Max pooling branch
        branch3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        
        # Concatenating the outputs from all branches
        output_tensor = Concatenate()([branch1, branch2, branch3])
        return output_tensor

    # Process through both blocks
    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    # Global average pooling
    gap = GlobalAveragePooling2D()(block2_output)
    
    # Fully connected layer for classification
    dense = Dense(units=128, activation='relu')(gap)
    dropout = Dropout(0.5)(dense)  # Adding dropout for regularization
    output_layer = Dense(units=10, activation='softmax')(dropout)  # CIFAR-10 has 10 classes

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model