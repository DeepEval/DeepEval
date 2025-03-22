import keras
from keras.layers import Input, Conv2D, Lambda, MaxPooling2D, Concatenate, Flatten, Dense, tf
from keras.models import Model

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def block1(input_tensor):
      x = Lambda(lambda x: tf.split(x, 2, axis=-1))(input_tensor)
      path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
      path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
      path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path1)
      path2 = x[1]
      output_tensor = Concatenate()([path1, path2])
      return output_tensor

    block1_output = block1(conv1)
    
    def block2(input_tensor):
      shape = Lambda(lambda x: tf.shape(x))(input_tensor)
      x = Lambda(lambda x: tf.reshape(x, [-1, shape[1], shape[2], int(shape[3] / 4)]) )(input_tensor)
      x = Lambda(lambda x: tf.transpose(x, [0, 1, 3, 2]) )(x)
      x = Lambda(lambda x: tf.reshape(x, [-1, shape[1], shape[2], shape[3]]) )(x)
      return x

    block2_output = block2(block1_output)

    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model