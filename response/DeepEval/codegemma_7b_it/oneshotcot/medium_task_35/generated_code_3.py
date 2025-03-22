import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, UpSampling2D, Add

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Stage 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    # Stage 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Additional convolutional and dropout layers
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling2)
    dropout1 = Dropout(rate=0.25)(conv3)

    # UpSampling and skip connections
    upsample1 = UpSampling2D(size=(2, 2))(dropout1)
    skip_connection1 = Add()([upsample1, conv2])

    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(skip_connection1)

    upsample2 = UpSampling2D(size=(2, 2))(conv4)
    skip_connection2 = Add()([upsample2, conv1])

    conv5 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(skip_connection2)

    # Output layer
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(conv5)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model