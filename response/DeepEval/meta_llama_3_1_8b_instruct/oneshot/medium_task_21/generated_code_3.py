import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the 1x1 convolution branch
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(0.2)(conv1)

    # Define the 1x1 convolution followed by 3x3 convolution branch
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    dropout2 = Dropout(0.2)(conv3)

    # Define the 1x1 convolution followed by two consecutive 3x3 convolutions branch
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)
    conv6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
    dropout3 = Dropout(0.2)(conv6)

    # Define the average pooling followed by 1x1 convolution branch
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_layer)
    conv7 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pool)
    dropout4 = Dropout(0.2)(conv7)

    # Concatenate the outputs from all branches
    output_tensor = Concatenate()([dropout1, dropout2, dropout3, dropout4])

    # Apply batch normalization
    bath_norm = BatchNormalization()(output_tensor)

    # Flatten the output
    flatten_layer = Flatten()(bath_norm)

    # Define the fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout5 = Dropout(0.2)(dense1)
    dense2 = Dense(units=64, activation='relu')(dropout5)
    dropout6 = Dropout(0.2)(dense2)
    output_layer = Dense(units=10, activation='softmax')(dropout6)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model