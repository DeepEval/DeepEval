import keras
from keras.layers import Input, SeparableConv2D, Conv2D, Lambda, Concatenate, Flatten, Dense, Add
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def main_path(input_tensor):
        # Split input into three groups along the channel
        split_layers = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(split_layers[0])
        path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(split_layers[1])
        path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(split_layers[2])
        
        concatenated = Concatenate()([path1, path2, path3])
        
        return concatenated

    # Branch path
    def branch_path(input_tensor):
        return Conv2D(filters=96, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)

    main_output = main_path(input_layer)
    branch_output = branch_path(input_layer)

    # Fuse both paths
    fused_output = Add()([main_output, branch_output])

    # Flatten and add dense layers
    flatten_layer = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Build model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model