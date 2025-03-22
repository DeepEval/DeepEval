import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Block 1

    # Path 1
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(conv1)
    pool1_flat = Flatten()(pool1)
    dropout1 = Dropout(0.2)(pool1_flat)

    # Path 2
    conv2 = Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    pool2_flat = Flatten()(pool2)
    dropout2 = Dropout(0.2)(pool2_flat)

    # Path 3
    conv3 = Conv2D(filters=32, kernel_size=(4, 4), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(conv3)
    pool3_flat = Flatten()(pool3)
    dropout3 = Dropout(0.2)(pool3_flat)

    # Concatenate the outputs of the three paths
    concatenated = Concatenate()([dropout1, dropout2, dropout3])

    # Block 2

    # Branch 1
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    pool4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(conv4)
    dropout4 = Dropout(0.2)(pool4)

    # Branch 2
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concatenated)
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same')(conv5)
    dropout5 = Dropout(0.2)(pool5)

    # Branch 3
    conv6 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(concatenated)
    pool6 = MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same')(conv6)
    dropout6 = Dropout(0.2)(pool6)

    # Branch 4
    pool7 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(concatenated)
    dropout7 = Dropout(0.2)(pool7)

    # Concatenate the outputs of the four branches
    concatenated_outputs = Concatenate()([dropout4, dropout5, dropout6, dropout7])

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concatenated_outputs)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model