import keras
from keras.layers import Input, Lambda, DepthwiseConv2D, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    def group_processing(input_tensor):
        group1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor[:, :, :, 0:1])
        group2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor[:, :, :, 1:2])
        group3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor[:, :, :, 2:3])
        output_tensor = Concatenate()([group1, group2, group3])
        return output_tensor
    
    split_layer = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    group1_output = group_processing(split_layer[0])
    group2_output = group_processing(split_layer[1])
    group3_output = group_processing(split_layer[2])
    fused_features = Concatenate()([group1_output, group2_output, group3_output])
    flattened = Flatten()(fused_features)
    output_layer = Dense(units=10, activation='softmax')(flattened)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model