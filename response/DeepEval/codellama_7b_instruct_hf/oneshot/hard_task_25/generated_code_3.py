import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    # Branch path
    branch_input = Input(shape=(32, 32, 3))
    branch_x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_input)
    branch_x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch_x)
    branch_x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_x)
    branch_x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch_x)
    branch_x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_x)
    branch_x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch_x)

    # Concatenate main and branch path outputs
    x = Concatenate()([x, branch_x])

    # Add a 1x1 convolutional layer
    x = Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    # Flatten and fully connected layers
    x = Flatten()(x)
    x = Dense(units=10, activation='softmax')(x)

    # Create and return the model
    model = keras.Model(inputs=[input_layer, branch_input], outputs=x)
    return model