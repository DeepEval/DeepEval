import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Step 2: Split the input image into three groups along the channel dimension
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Step 3: Define the block for multi-scale feature extraction
    def multi_scale_block(channels):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(channels[0])
        path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(channels[1])
        path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(channels[2])
        
        # Step 4: Concatenate the outputs from the three paths
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor

    # Apply the multi-scale block to the split channels
    block_output = multi_scale_block(split_channels)
    
    # Step 5: Flatten the concatenated output
    flatten_layer = Flatten()(block_output)
    
    # Step 6: Add dense layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model