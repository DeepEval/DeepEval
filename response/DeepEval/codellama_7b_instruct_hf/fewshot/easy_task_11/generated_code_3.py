import keras
from keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Dropout, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Average pooling layer with 5x5 window and 3x3 stride
    pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(input_layer)

    # 1x1 convolutional layer to enhance depth and introduce nonlinearity
    conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool)

    # Flatten the feature maps
    flatten = Flatten()(conv)

    # Two fully connected layers with dropout
    dense1 = Dense(units=128, activation='relu')(flatten)
    dropout1 = Dropout(rate=0.2)(dense1)
    dense2 = Dense(units=64, activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.2)(dense2)

    # Final output layer with softmax activation
    output_layer = Dense(units=10, activation='softmax')(dropout2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model