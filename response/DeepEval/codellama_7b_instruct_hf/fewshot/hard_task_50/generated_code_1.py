import keras
from keras.layers import Input, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Reshape, DepthwiseConv2D
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first block
    maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    flatten1 = Flatten()(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    flatten2 = Flatten()(maxpool2)
    maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    flatten3 = Flatten()(maxpool3)
    concatenated = Concatenate()([flatten1, flatten2, flatten3])
    dropout = Dropout(0.2)(concatenated)

    # Define the second block
    inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=4, axis=-1))(dropout)
    conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
    conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
    conv4 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(inputs_groups[3])
    concatenated_conv = Concatenate()([conv1, conv2, conv3, conv4])

    # Define the output layer
    flatten = Flatten()(concatenated_conv)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model