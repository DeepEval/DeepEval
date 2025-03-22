import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    conv4 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv3)

    # Restore the number of channels to match the input
    conv5 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv4)

    # Add the original input to the processed features
    output_tensor = keras.layers.Add()([conv5, input_layer])

    # Apply dropout to mitigate overfitting
    dropout = Dropout(rate=0.25)(output_tensor)

    # Flatten the output
    flatten_layer = Flatten()(dropout)

    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model