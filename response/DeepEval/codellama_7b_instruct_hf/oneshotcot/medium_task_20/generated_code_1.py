import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # First path: 1x1 convolution
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second path: 2x2 convolution
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)

    # Third path: 3x3 convolution
    path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)

    # Fourth path: Max pooling followed by 1x1 convolution
    path4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)

    # Concatenate the outputs of all paths
    output_tensor = Concatenate()([path1, path2, path3, path4])

    # Batch normalization
    output_tensor = BatchNormalization()(output_tensor)

    # Flatten the output
    output_tensor = Flatten()(output_tensor)

    # Dense layer with 128 units
    output_tensor = Dense(units=128, activation='relu')(output_tensor)

    # Dense layer with 10 units (for classification)
    output_tensor = Dense(units=10, activation='softmax')(output_tensor)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_tensor)

    return model