import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda


def dl_model():  

    input_tensor = Input(shape=(32, 32, 3)) 

    # Main Path
    split_tensor = Lambda(lambda x: tf.split(x, 3, axis=1))(input_tensor)  

    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
    conv3_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
    conv5_1 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])

    main_path_output = Concatenate(axis=1)([conv1_1, conv3_1, conv5_1])

    # Branch Path
    branch_conv = Conv2D(filters=32 * 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)  

    # Concatenate and Final Layers
    output = layers.add([main_path_output, branch_conv])
    
    x = Flatten()(output)
    x = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output_layer)

    return model