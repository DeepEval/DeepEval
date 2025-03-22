import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Specialized block for capturing local features
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    dropout = Dropout(rate=0.2)(maxpool)

    # Block for additional processing
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout)
    conv4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)
    dropout2 = Dropout(rate=0.2)(maxpool2)

    # Global average pooling layer
    avgpool = GlobalAveragePooling2D()(dropout2)

    # Flattening layer and fully connected layer
    flatten = Flatten()(avgpool)
    dense = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model