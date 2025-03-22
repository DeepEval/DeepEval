import keras
from keras.layers import Input, Lambda, DepthwiseConv2D, BatchNormalization, Flatten, Concatenate, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
    conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
    output_tensor = Concatenate()([conv1, conv2, conv3])
    batch_norm = BatchNormalization()(output_tensor)

    # Second block
    branch1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(batch_norm)
    branch2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm)
    branch3 = DepthwiseConv2D(kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(batch_norm)
    branch4 = DepthwiseConv2D(kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(batch_norm)
    branch5 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm)
    output_tensor = Concatenate()([branch1, branch2, branch3, branch4, branch5])
    flatten = Flatten()(output_tensor)
    dense1 = Dense(units=64, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model