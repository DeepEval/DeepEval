import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Lambda, Reshape
from tensorflow import tf

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # First block: Average pooling layers
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    flat_pool1 = Flatten()(pool1)
    flat_pool2 = Flatten()(pool2)
    flat_pool3 = Flatten()(pool3)

    concat_pool = Concatenate()([flat_pool1, flat_pool2, flat_pool3])

    dense1 = Dense(units=128, activation='relu')(concat_pool)
    reshape_layer = Reshape((1, 128))(dense1)

    # Second block: Depthwise separable convolutions
    def split_and_process(input_tensor):
      split_tensor = Lambda(lambda x: tf.split(x, 4, axis=1))(input_tensor)
      conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', 
                    activation='relu')(split_tensor[0])
      conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                    activation='relu')(split_tensor[1])
      conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same',
                    activation='relu')(split_tensor[2])
      conv4 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same',
                    activation='relu')(split_tensor[3])
      return Concatenate()([conv1, conv2, conv3, conv4])

    processed_tensor = split_and_process(reshape_layer)
    flattened = Flatten()(processed_tensor)
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model