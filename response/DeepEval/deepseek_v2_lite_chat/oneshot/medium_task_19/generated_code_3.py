import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Branches
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer) # Dimensionality reduction
    branch2 = Conv2D(filters=64, kernel_size=(1, 1) + (3, 3), padding='same')(input_layer) # 1x1 convolution followed by 3x3 convolution
    branch3 = Conv2D(filters=64, kernel_size=(1, 1) + (5, 5), padding='same')(input_layer) # 1x1 convolution followed by 5x5 convolution
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid')(input_layer) # 3x3 max pooling

    # Processing the branches
    branch1 = BatchNormalization()(branch1)
    branch2 = BatchNormalization()(branch2)
    branch3 = BatchNormalization()(branch3)
    branch4 = BatchNormalization()(branch4)

    # Concatenate the outputs of the branches
    concat_layer = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])

    # Flatten the concatenated layer
    flat = Flatten()(concat_layer)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()
model.summary()