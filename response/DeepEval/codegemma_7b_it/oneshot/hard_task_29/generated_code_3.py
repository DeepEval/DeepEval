import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Main Path and Branch Path

    def block1(input_tensor):

        # Main Path
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

        # Branch Path
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)

        # Combine outputs through addition
        output_tensor = keras.layers.Add()([conv2, conv4])

        return output_tensor

    block1_output = block1(input_tensor=input_layer)
    bath_norm1 = BatchNormalization()(block1_output)

    # Block 2: Max Pooling Layers

    def block2(input_tensor):

        maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_tensor)
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_tensor)

        # Flatten outputs and concatenate
        flatten_outputs = [keras.layers.Flatten()(x) for x in [maxpool1, maxpool2, maxpool3]]
        concat_output = Concatenate()(flatten_outputs)

        return concat_output

    block2_output = block2(input_tensor=bath_norm1)

    # Fully Connected Layers

    dense1 = Dense(units=128, activation='relu')(block2_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model Definition

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model