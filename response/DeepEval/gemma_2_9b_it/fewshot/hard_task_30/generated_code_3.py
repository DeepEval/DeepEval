import keras
from keras.layers import Input, Conv2D, Lambda, MaxPooling2D, Concatenate, DepthwiseConv2D, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Dual-path structure
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path) 

    branch_path = input_layer

    combined_output = Add()([main_path, branch_path])

    # Block 2: Channel Splitting and Depthwise Separable Convolutions
    split_tensor = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(combined_output)

    conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
    conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])

    output_tensor = Concatenate()([conv1, conv2, conv3])

    # Fully connected layers
    flatten_layer = Flatten()(output_tensor)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model