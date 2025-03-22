import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(224, 224, 3))

    # First Sequential Feature Extraction Layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)

    # Second Sequential Feature Extraction Layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(pool1)
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)

    # Additional Convolutional Layers
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(pool2)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(conv3)
    conv5 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu')(conv4)

    # Second Average Pooling Layer
    pool3 = AveragePooling2D(pool_size=(2, 2))(conv5)

    # Flatten Layer
    flatten = Flatten()(pool3)

    # Fully Connected Layers with Dropout
    dense1 = Dense(units=1024, activation='relu')(flatten)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(units=512, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)

    # Output Layer
    output_layer = Dense(units=1000, activation='softmax')(dropout2)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model