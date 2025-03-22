import keras
from keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    def conv_block(input_tensor):
      conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
      conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
      conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
      return conv3

    split_tensor = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=2))(input_layer)
    group1_output = conv_block(split_tensor[0])
    group2_output = conv_block(split_tensor[1])
    group3_output = conv_block(split_tensor[2])

    main_path = Add()([group1_output, group2_output, group3_output])
    combined_features = Add()([main_path, input_layer])

    flatten_layer = Flatten()(combined_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model