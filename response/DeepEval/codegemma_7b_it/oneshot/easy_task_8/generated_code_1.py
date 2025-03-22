import keras
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Depthwise Separable Convolutional Layer
    conv_dw = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', depthwise_initializer='he_normal', activation='relu')(input_layer)

    # 1x1 Convolutional Layer for Feature Extraction
    conv_pw = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_dw)

    # Dropout to Mitigate Overfitting
    dropout = Dropout(0.25)(conv_pw)

    # Max Pooling Layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(dropout)

    # Flatten the Output
    flatten_layer = Flatten()(max_pooling)

    # Fully Connected Layer for Classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)

    # Output Layer for Probabilities
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model