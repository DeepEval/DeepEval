import keras
from keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Average pooling layer with 5x5 window and 3x3 stride
    pooling = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(input_layer)

    # 1x1 convolutional layer
    conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pooling)

    # Flatten the feature maps
    flatten = Flatten()(conv)

    # Two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Dropout layer to mitigate overfitting
    dropout = Dropout(rate=0.5)(dense2)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=dropout)
    return model