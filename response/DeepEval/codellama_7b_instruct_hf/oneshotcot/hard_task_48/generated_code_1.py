import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    # Split input into three groups
    group1 = Lambda(lambda x: x[:, :, :, 0:16])(input_layer)
    group2 = Lambda(lambda x: x[:, :, :, 16:32])(input_layer)
    group3 = Lambda(lambda x: x[:, :, :, 32:48])(input_layer)

    # Feature extraction using separable convolutional layers with different kernel sizes
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group2)
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(group3)

    # Batch normalization
    norm1 = BatchNormalization()(conv1)
    norm2 = BatchNormalization()(conv2)
    norm3 = BatchNormalization()(conv3)

    # Concatenate the outputs of the three groups
    concat = Concatenate()([norm1, norm2, norm3])

    # Block 2
    # Four parallel branches
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat)
    path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(concat)
    path4 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(concat)

    # Concatenate the outputs of the four paths
    concat = Concatenate()([path1, path2, path3, path4])

    # Flatten the output
    flat = Flatten()(concat)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model