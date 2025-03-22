from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, \
    MaxPooling2D, concatenate, Flatten, Dense

def dl_model():
    # Define input layer
    inputs = Input(shape=(32, 32, 64))

    # Compress input channels
    compress = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(inputs)
    compress = BatchNormalization()(compress)
    compress = Activation('relu')(compress)

    # Expand features through parallel convolutional layers
    expand_1x1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(compress)
    expand_1x1 = BatchNormalization()(expand_1x1)
    expand_1x1 = Activation('relu')(expand_1x1)

    expand_3x3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(compress)
    expand_3x3 = BatchNormalization()(expand_3x3)
    expand_3x3 = Activation('relu')(expand_3x3)

    # Concatenate results
    concat = concatenate([expand_1x1, expand_3x3])

    # Flatten and fully connected layers
    flatten = Flatten()(concat)
    dense_1 = Dense(units=128, activation='relu')(flatten)
    dense_2 = Dense(units=2, activation='softmax')(dense_1)

    # Create model
    model = Model(inputs=inputs, outputs=dense_2)

    return model