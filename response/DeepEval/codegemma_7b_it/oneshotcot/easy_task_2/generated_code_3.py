import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():     

    input_layer = Input(shape=(224, 224, 3))

    # First feature extraction layer
    conv1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    # Second feature extraction layer
    conv2 = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool1)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Third convolutional layer
    conv3 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool2)

    # Fourth convolutional layer
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)

    # Fifth convolutional layer
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)

    # Sixth convolutional layer and average pooling layer
    conv6 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
    avg_pool3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv6)

    # Flatten the feature maps
    flatten_layer = Flatten()(avg_pool3)

    # First fully connected layer with dropout
    dense1 = Dense(units=4096, activation='relu')(flatten_layer)
    dropout1 = Dropout(rate=0.4)(dense1)

    # Second fully connected layer with dropout
    dense2 = Dense(units=4096, activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.4)(dense2)

    # Output layer with softmax activation
    output_layer = Dense(units=1000, activation='softmax')(dropout2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model