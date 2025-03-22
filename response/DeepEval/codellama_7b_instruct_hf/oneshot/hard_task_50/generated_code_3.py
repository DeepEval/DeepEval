import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras.models import Model


def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool2)
    maxpool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)

    # Flatten the output of the first block
    flatten_layer = Flatten()(maxpool3)

    # Define the second block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(flatten_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)

    # Concatenate the outputs of the second block
    concat_layer = Concatenate()([conv1, conv2, conv3])

    # Apply batch normalization and flatten the output
    batch_norm = BatchNormalization()(concat_layer)
    flatten_layer = Flatten()(batch_norm)

    # Define the third block
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    dense3 = Dense(units=32, activation='relu')(dense2)

    # Define the output layer
    output_layer = Dense(units=10, activation='softmax')(dense3)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model