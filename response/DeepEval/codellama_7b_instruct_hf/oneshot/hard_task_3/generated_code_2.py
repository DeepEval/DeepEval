import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)

    # Main pathway
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)
    concatenate_layer = Concatenate()([conv1, conv2, conv3, maxpool])

    # Branch pathway
    branch_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    branch_conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    branch_conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    branch_maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_conv3)
    branch_concatenate_layer = Concatenate()([branch_conv1, branch_conv2, branch_conv3, branch_maxpool])

    # Combine the outputs from both pathways using addition
    output = concatenate_layer + branch_concatenate_layer

    # Flatten and pass through fully connected layers
    flatten_layer = Flatten()(output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model