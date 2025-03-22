import keras
from keras.layers import Input, Lambda, DepthwiseConv2D, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    x = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])
    main_path = Concatenate()([conv1, conv2, conv3])

    # Branch Path
    branch_path = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Add Main and Branch Paths
    added_output = keras.layers.add([main_path, branch_path])

    # Fully Connected Layers
    flatten_layer = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model