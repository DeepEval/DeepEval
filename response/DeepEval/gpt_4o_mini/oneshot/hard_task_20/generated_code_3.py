import keras
from keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense, Concatenate
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have shape 32x32x3

    # Main path
    split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Convolutional layers for each split
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_tensor[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_tensor[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_tensor[2])

    # Concatenate the outputs of the main path
    main_output = Concatenate()([conv1, conv2, conv3])

    # Branch path
    branch_output = Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)  # Align channels

    # Fuse features from main and branch paths
    fused_output = Add()([main_output, branch_output])

    # Flatten and classification layers
    flatten_layer = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model