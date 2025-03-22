import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, GlobalAveragePooling2D, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First block: Splitting and applying separable convolutions
    def separable_conv_block(input_tensor):
        # Split input tensor into 3 groups
        split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply separable convolutions with different kernel sizes
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu', depthwise=True)(split_tensors[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', depthwise=True)(split_tensors[1])
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', depthwise=True)(split_tensors[2])
        
        # Concatenate the outputs from the three branches
        return Concatenate()([conv1, conv2, conv3])
    
    block1_output = separable_conv_block(input_layer)

    # Second block: Multiple branches for enhanced feature extraction
    def feature_extraction_block(input_tensor):
        # Branch 1: 3x3 Convolution
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        
        # Branch 2: Series of 1x1 and 3x3 convolutions
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        
        # Branch 3: Max Pooling
        branch3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        
        # Concatenate outputs from all branches
        return Concatenate()([branch1, branch2, branch3])
    
    block2_output = feature_extraction_block(block1_output)

    # Global average pooling and final classification layer
    global_avg_pooling = GlobalAveragePooling2D()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(global_avg_pooling)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model