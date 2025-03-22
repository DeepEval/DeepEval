import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():     
    input_layer = Input(shape=(224, 224, 3))

    # First Feature Extraction Layer
    conv_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    avg_pool_1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_1)

    # Second Feature Extraction Layer
    conv_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool_1)
    avg_pool_2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_2)

    # Additional Convolutional Layers and Pooling
    conv_3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool_2)
    conv_4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_3)
    conv_5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_4)
    avg_pool_3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_5)

    # Flatten Feature Maps
    flatten_layer = Flatten()(avg_pool_3)

    # Fully Connected Layers with Dropout
    dense_1 = Dense(units=4096, activation='relu')(flatten_layer)
    dropout_1 = Dropout(rate=0.5)(dense_1)
    dense_2 = Dense(units=4096, activation='relu')(dropout_1)
    dropout_2 = Dropout(rate=0.5)(dense_2)

    # Output Layer
    output_layer = Dense(units=1000, activation='softmax')(dropout_2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model