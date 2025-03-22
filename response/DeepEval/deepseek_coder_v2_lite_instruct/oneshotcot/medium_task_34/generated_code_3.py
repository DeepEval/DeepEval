import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, BatchNormalization, Flatten, Dense, Conv2DTranspose
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First part: Sequential convolutional and max-pooling layers
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Second part: Processing through convolutional, dropout, and another convolutional layer
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    # Store the output for skip connections
    conv_output = x

    # Third part: Upsampling through convolutional and transposed convolutional layers with skip connections
    x = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Concatenate()([x, conv_output])
    conv_output = x
    x = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Concatenate()([x, conv_output])
    conv_output = x
    x = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Concatenate()([x, conv_output])

    # Final part: 1x1 convolutional layer for classification
    x = Conv2D(filters=10, kernel_size=(1, 1), activation='sigmoid')(x)
    output_layer = Flatten()(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model