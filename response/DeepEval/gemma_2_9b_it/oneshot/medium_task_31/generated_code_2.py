import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    def split_and_conv(input_tensor):
      split_tensor = Lambda(lambda x: tf.split(x, 3, axis=2))(input_tensor)
      conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
      conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
      conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
      return Concatenate(axis=3)([conv1, conv2, conv3])

    x = split_and_conv(input_layer)
    x = Flatten()(x)
    dense1 = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model