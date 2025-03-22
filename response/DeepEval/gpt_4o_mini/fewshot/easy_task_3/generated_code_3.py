import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First feature extraction block
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv1_2)

    # Second feature extraction block
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool1)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2_2)

    # Third feature extraction block
    conv3_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool2)
    conv3_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_1)
    conv3_3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_2)
    max_pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv3_3)

    # Fourth feature extraction block
    conv4_1 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool3)
    conv4_2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4_1)
    conv4_3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4_2)
    max_pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv4_3)

    # Flattening the output from the last convolutional block
    flatten_layer = Flatten()(max_pool4)

    # Fully connected layers
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dropout1 = Dropout(0.5)(dense1)  # Dropout for regularization
    dense2 = Dense(units=256, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)  # Dropout for regularization
    output_layer = Dense(units=10, activation='softmax')(dropout2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model