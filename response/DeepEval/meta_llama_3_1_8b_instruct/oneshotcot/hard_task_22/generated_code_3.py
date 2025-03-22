import keras
from keras.layers import Input, Lambda, Conv2D, Concatenate, Add, BatchNormalization, Flatten, Dense, SeparableConv2D
from keras import backend as K
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def main_path(input_tensor):
        # Split the input along the channel dimension
        x1, x2, x3 = tf.split(input_tensor, num_or_size_splits=3, axis=3)
        
        # Apply separable convolutional layers of varying kernel sizes
        path1 = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x1)
        path2 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x2)
        path3 = SeparableConv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(x3)
        
        # Concatenate the outputs from the groups
        output_tensor = Concatenate()([path1, path2, path3])
        
        return output_tensor
    
    # Apply the main path
    main_output = main_path(input_layer)

    # Branch path
    branch_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse the outputs from both paths
    fused_output = Add()([main_output, branch_output])

    # Apply batch normalization
    batch_norm = BatchNormalization()(fused_output)

    # Flatten the output
    flatten_layer = Flatten()(batch_norm)

    # Apply two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model