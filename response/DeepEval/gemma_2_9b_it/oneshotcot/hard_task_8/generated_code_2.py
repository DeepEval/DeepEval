import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Permute

def dl_model():     

        input_layer = Input(shape=(28, 28, 1))

        # Block 1
        x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
        x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

        branch_x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
        branch_x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_x)

        x = Concatenate()([x, branch_x])

        # Block 2
        shape = keras.layers.Lambda(lambda x: keras.backend.shape(x))(x)
        x = Reshape(target_shape=(shape[1], shape[2], 4, 32))(x)
        x = Permute(axes=(0, 1, 3, 2))(x)
        x = Reshape(target_shape=(shape[1], shape[2], 128))(x)

        # Flatten and output layer
        x = Flatten()(x)
        output_layer = Dense(units=10, activation='softmax')(x)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model