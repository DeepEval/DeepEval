import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, Add, Flatten, Dense

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path: Split the input into three groups
    split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Feature extraction for each group with different kernel sizes
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
    path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])

    # Concatenate the outputs from the main path
    main_path_output = Concatenate()([path1, path2, path3])

    # Branch path: 1x1 convolution to align output channels
    branch_path_output = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the main and branch paths through addition
    fused_features = Add()([main_path_output, branch_path_output])

    # Flatten and classify
    flatten_layer = Flatten()(fused_features)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model