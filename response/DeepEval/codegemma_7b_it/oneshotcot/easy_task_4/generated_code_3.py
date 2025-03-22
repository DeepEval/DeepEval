import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # First block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(conv2)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)

    # Second block
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(max_pooling1)
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(conv4)
    conv6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(conv5)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv6)

    # Feature extraction
    feature_maps = max_pooling2

    # Feature flattening
    flatten_layer = Flatten()(feature_maps)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model creation
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model