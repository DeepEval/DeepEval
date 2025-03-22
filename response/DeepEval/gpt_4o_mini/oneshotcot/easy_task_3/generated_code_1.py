import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # First block: Convolution, Convolution, MaxPooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    max_pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Second block: Convolution, Convolution, MaxPooling
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(max_pool1)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv3)
    max_pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Third block: Convolution, Convolution, Convolution, MaxPooling
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(max_pool2)
    conv6 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv5)
    conv7 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv6)
    max_pool3 = MaxPooling2D(pool_size=(2, 2))(conv7)

    # Fourth block: Convolution, Convolution, Convolution, MaxPooling
    conv8 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(max_pool3)
    conv9 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(conv8)
    conv10 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(conv9)
    max_pool4 = MaxPooling2D(pool_size=(2, 2))(conv10)

    # Flattening the feature maps
    flatten_layer = Flatten()(max_pool4)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout1 = Dropout(0.5)(dense1)  # Adding dropout for regularization
    dense2 = Dense(units=64, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)  # Adding dropout for regularization
    output_layer = Dense(units=10, activation='softmax')(dropout2)

    # Creating the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model