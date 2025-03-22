import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def main_path(input_tensor):
        # Split the input into three groups along the channel axis
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply different convolutions on each split
        conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
        conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
        conv_5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(splits[2])
        
        # Concatenate the results
        concatenated = Concatenate()([conv_1x1, conv_3x3, conv_5x5])
        
        return concatenated

    main_output = main_path(input_layer)

    # Branch path
    branch_output = Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine main path and branch path
    fused_output = Add()([main_output, branch_output])

    # Flatten the result and add fully connected layers for classification
    flatten_layer = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model