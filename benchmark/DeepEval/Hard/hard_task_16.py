import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Lambda, Concatenate, Dense, GlobalMaxPooling2D, Reshape, Multiply, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    def block_1(input_tensor):

        groups = Lambda(function=lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)

        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(groups[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
        group_0 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid')(conv2)

        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(groups[1])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        group_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(conv2)

        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(groups[2])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        group_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(conv2)

        output_tensor = Concatenate()([group_0, group_1, group_2])

        return output_tensor
    
    def block_2(input_tensor):
        
        squeeze = GlobalMaxPooling2D()(input_tensor)
    
        input_channels = input_tensor.shape[-1]
        excitation = Dense(units=input_channels // 16, activation='relu')(squeeze)
        excitation = Dense(units=input_channels, activation='sigmoid')(excitation)
        excitation = Reshape(target_shape=(1, 1, input_channels))(excitation)
        output_tensor = Multiply()([input_tensor, excitation])

        return output_tensor
    
    block_1_output = block_1(input_tensor=input_layer)
    transition_conv = Conv2D(filters=input_layer.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(block_1_output)
    block_2_output = block_2(input_tensor=transition_conv)

    added = Add()([block_2_output, input_layer])

    output_layer = Dense(units=10, activation='softmax')(added)
    model = Model(inputs=input_layer, outputs=output_layer)

    return model
