import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Lambda, DepthwiseConv2D, Concatenate, Dense, Flatten, Add

def dl_model():
    input_layer = Input(shape=(32,32,3))
    
    def feature_extraction(input_tensor, kernel_size):
        conv = DepthwiseConv2D(kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv

    split_input = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    main_path_outputs = []
    for kernel_size in [1, 3, 5]:
        main_path_outputs.append(feature_extraction(split_input[0], kernel_size))
    main_path_output = Concatenate()([main_path_outputs[0], main_path_outputs[1], main_path_outputs[2]])
    
    branch_path_output = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_input[1])
    
    merged_output = Add()([main_path_output, branch_path_output])
    
    flatten_output = Flatten()(merged_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_output)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model