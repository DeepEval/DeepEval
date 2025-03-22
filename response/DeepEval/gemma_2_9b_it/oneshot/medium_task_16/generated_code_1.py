import keras
from keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    def split_and_conv(input_tensor):
        split_tensor = Lambda(lambda x: tf.split(x, 3, axis=2))(input_tensor)
        conv1 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
        conv2 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
        conv3 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
        return [conv1, conv2, conv3]

    split_output = split_and_conv(input_layer)
    pool1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(split_output[0])
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(split_output[1])
    pool3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(split_output[2])
    
    concat_output = Concatenate(axis=3)([pool1, pool2, pool3])
    flatten_layer = Flatten()(concat_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model