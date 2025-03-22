import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Lambda, Add, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor):

        num_channels = input_tensor.shape[-1]
        
        num_groups = 3
        channels_per_group = num_channels // num_groups
        groups = Lambda(function=lambda x: tf.split(value=x, num_or_size_splits=num_groups, axis=-1))(input_tensor)

        path_1 = groups[0]
        path_2 = Conv2D(filters=channels_per_group, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(groups[1])
        added = Add()([groups[2], path_2])
        path_3 = Conv2D(filters=channels_per_group, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(added)
        main_path = Concatenate()([path_1, path_2, path_3])
        
        branch_path = Conv2D(filters=num_channels, kernel_size=(1, 1), strides=(1,1), padding='same',activation='relu')(input_tensor)

        output_tensor = Add()([main_path, branch_path])

        return output_tensor
    
    block_output = block(input_tensor=input_layer)
    flatten_output = Flatten()(block_output)
    dense = Dense(units=128, activation='relu')(flatten_output)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
