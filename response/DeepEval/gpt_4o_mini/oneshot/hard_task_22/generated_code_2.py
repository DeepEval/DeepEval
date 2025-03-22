import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense, Add
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Main path processing
    def main_path(split_input):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu', depthwise=True)(split_input)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', depthwise=True)(split_input)
        path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', depthwise=True)(split_input)
        return path1, path2, path3
    
    main_outputs = [main_path(split_input) for split_input in split_inputs]
    
    # Concatenate the outputs of the main path
    concat_main = Concatenate()([*main_outputs[0], *main_outputs[1], *main_outputs[2]])
    
    # Branch path for aligning channels
    branch_output = Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse the outputs from both paths
    fused_output = Add()([concat_main, branch_output])
    
    # Flatten and fully connected layers
    flatten_layer = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model