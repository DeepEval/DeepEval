import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Lambda, Concatenate, Add, Dense


def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    def block(input_tensor):

        inputs_groups = Lambda(function=lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        main_path = Concatenate()([conv1, conv2, conv3])
       
        branch_path = Conv2D(filters=main_path.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Add()([main_path, branch_path])

        return output_tensor
    
    block_output = block(input_tensor=input_layer)
    dense_1_output = Dense(units=128, activation='relu')(block_output)
    output_layer = Dense(units=10, activation='softmax')(dense_1_output)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model




