import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.layers import Lambda, Reshape
from keras.layers.core import Activation
from keras.layers.merge import Add
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.layers import GlobalMaxPooling2D
from keras import regularizers
from tensorflow.keras import backend as K
import tensorflow as tf

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def split(input_tensor):
        return tf.split(input_tensor, num_or_size_splits=3, axis=-1)

    split_layer = Lambda(split)(input_layer)
    
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    path1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path1)

    path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)

    path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[2])
    path3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path3)

    concat_layer = Concatenate()([path1, path2, path3])

    # Transition Convolution
    transition_conv = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_layer)

    # Block 2
    global_max_pooling = GlobalMaxPooling2D()(transition_conv)

    # weights generation
    weights_1 = Dense(16, activation='relu')(global_max_pooling)
    weights_2 = Dense(16, activation='relu')(weights_1)

    # Reshape weights
    reshape_weights = Reshape((16, 16))(weights_2)

    # Element-wise multiply
    main_path_output = Multiply()([transition_conv, reshape_weights])

    # Branch
    branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Add main path output and branch output
    add_output = Add()([main_path_output, branch])

    output_layer = Dense(units=10, activation='softmax')(Flatten()(add_output))

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model