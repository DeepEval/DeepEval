import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(224, 224, 3))

    # First feature extraction layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv1)

    # Second feature extraction layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2)

    # Additional convolutional layers
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)
    pool3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv5)

    # Flatten the feature maps
    flatten_layer = Flatten()(pool3)

    # Fully connected layers with dropout
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dropout1 = Dropout(rate=0.5)(dense1)
    
    dense2 = Dense(units=512, activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.5)(dense2)

    # Output layer for classification
    output_layer = Dense(units=1000, activation='softmax')(dropout2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model