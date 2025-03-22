import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Average

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv2)

    # Pooling layers with varying window sizes
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(conv3)

    # Flatten and concatenate the outputs
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)
    concatenated = Concatenate()(outputs=[flat1, flat2, flat3])

    # Fully connected layers
    dense1 = Dense(units=512, activation='relu')(concatenated)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()
model.summary()