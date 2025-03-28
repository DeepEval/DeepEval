import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():

    input_layer = Input(shape=(224, 224, 3))

    # Sequential Feature Extraction Layers
    conv1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid', activation='relu')(input_layer)
    avg_pool1 = AveragePooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv1)
    conv2 = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(avg_pool1)
    avg_pool2 = AveragePooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv2)

    # Additional Convolutional Layers
    conv3 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(avg_pool2)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv3)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv4)
    avg_pool3 = AveragePooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv5)

    # Flatten Feature Maps
    flatten_layer = Flatten()(avg_pool3)

    # Fully Connected Layers with Dropout
    dense1 = Dense(units=4096, activation='relu')(flatten_layer)
    dropout1 = Dropout(rate=0.4)(dense1)
    dense2 = Dense(units=4096, activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.4)(dense2)

    # Output Layer
    output_layer = Dense(units=1000, activation='softmax')(dropout2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model