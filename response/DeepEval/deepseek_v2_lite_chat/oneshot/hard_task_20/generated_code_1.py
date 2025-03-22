import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # Input shape for CIFAR-10 (32x32 pixels, 3 color channels)

    # Split the input into three groups
    group1, group2, group3 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)

    # Convolutional layers for the main path
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group2)
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(group3)

    # Concatenate the outputs of the main path
    concat = Concatenate(axis=-1)([conv1, conv2, conv3])

    # 1x1 convolutional layer for the branch path
    branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the outputs of the main and branch paths
    fused = keras.layers.Add()([concat, branch])

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(fused)
    flat = Flatten()(batch_norm)

    # Fully connected layers for classification
    dense1 = Dense(units=256, activation='relu')(flat)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate and return the model
model = dl_model()