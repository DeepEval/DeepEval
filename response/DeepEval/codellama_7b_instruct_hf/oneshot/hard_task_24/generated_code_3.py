import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications import VGG16

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the base model
    base_model = VGG16(include_top=False, input_shape=input_shape, pooling='avg')

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Define the new layers
    x = base_model.output
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dense(units=32, activation='relu')(x)
    x = Dense(units=10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=base_model.input, outputs=x)

    return model