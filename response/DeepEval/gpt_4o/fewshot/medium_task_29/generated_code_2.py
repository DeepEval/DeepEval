import keras
from keras.layers import Input, MaxPooling2D, Flatten, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB

    # First pooling layer with a 1x1 window
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    flatten1 = Flatten()(pool1)

    # Second pooling layer with a 2x2 window
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    flatten2 = Flatten()(pool2)

    # Third pooling layer with a 4x4 window
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    flatten3 = Flatten()(pool3)

    # Concatenate flattened outputs
    concatenated = Concatenate()([flatten1, flatten2, flatten3])

    # Fully connected layers for classification
    dense1 = Dense(units=256, activation='relu')(concatenated)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model