import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(224, 224, 3))

    # First Sequential Feature Extraction Layer
    conv_pool_1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    avg_pool_1 = AveragePooling2D(pool_size=(2, 2))(conv_pool_1)

    # Second Sequential Feature Extraction Layer
    conv_pool_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(avg_pool_1)
    avg_pool_2 = AveragePooling2D(pool_size=(2, 2))(conv_pool_2)

    # Additional Convolutional Layers
    conv_3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(avg_pool_2)
    conv_4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(conv_3)
    conv_5 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu')(conv_4)

    # Final Average Pooling Layer
    avg_pool_3 = AveragePooling2D(pool_size=(2, 2))(conv_5)

    # Flatten Layer
    flatten_layer = Flatten()(avg_pool_3)

    # Fully Connected Layers with Dropout
    dense_1 = Dense(units=256, activation='relu')(flatten_layer)
    dropout_1 = Dropout(0.5)(dense_1)
    dense_2 = Dense(units=128, activation='relu')(dropout_1)
    dropout_2 = Dropout(0.5)(dense_2)

    # Output Layer
    output_layer = Dense(units=1000, activation='softmax')(dropout_2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model