import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Lambda, Dropout, Add, SeparableConv2D, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    def block_1(input_tensor):

        conv1_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        dropout1 = Dropout(rate=0.5)(conv1_1)
        main_path = Conv2D(filters=input_tensor.shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout1)

        output_tensor = Add()([main_path, input_tensor])

        return output_tensor

    def block_2(input_tensor):

        inputs_groups = Lambda(function=lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        path1= Dropout(rate=0.5)(conv1)
        conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        path2= Dropout(rate=0.5)(conv2)
        conv3 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        path3 = Dropout(rate=0.5)(conv3)

        output_tensor = Concatenate()([path1, path2, path3])

        return output_tensor

    block_1_output = block_1(input_tensor=input_layer)
    block_2_output = block_2(input_tensor=block_1_output)

    flatten = Flatten()(block_2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = Model(inputs=input_layer, outputs=output_layer)

    return model
