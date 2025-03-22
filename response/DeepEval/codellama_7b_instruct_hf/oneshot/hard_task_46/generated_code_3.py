import keras
from keras.layers import Input, Lambda, Conv2D, SeparableConv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split input into three groups along the channel dimension
    input_split = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)

    # Define first block
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_split[0])
    conv2 = SeparableConv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_split[1])
    conv3 = SeparableConv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_split[2])
    conv_output = Concatenate()([conv1, conv2, conv3])
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_output)

    # Define second block
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling)
    branch4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling)
    branch_output = Concatenate()([branch1, branch2, branch3, branch4])

    # Define global average pooling and fully connected layers
    pooling = GlobalAveragePooling2D()(branch_output)
    flatten = Flatten()(pooling)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Define and return model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model