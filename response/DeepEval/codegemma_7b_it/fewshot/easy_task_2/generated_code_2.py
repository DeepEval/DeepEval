import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(224, 224, 3))

    # Feature Extraction Block 1
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

    # Feature Extraction Block 2
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool1)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    # Feature Extraction Block 3
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool2)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)
    avg_pool3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv5)

    # Flatten and Fully Connected Layers
    flatten = Flatten()(avg_pool3)
    dense1 = Dense(units=512, activation='relu')(flatten)
    dropout1 = Dropout(rate=0.5)(dense1)
    dense2 = Dense(units=1024, activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.5)(dense2)

    # Output Layer
    output_layer = Dense(units=1000, activation='softmax')(dropout2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model