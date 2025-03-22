import keras
from keras.layers import Input, Conv2D, AvgPool2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(224, 224, 3))

    # Sequential feature extraction layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    avg_pool1 = AvgPool2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(avg_pool1)
    avg_pool2 = AvgPool2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Three additional convolutional layers
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(avg_pool2)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv3)
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv4)

    # Another average pooling layer to reduce dimensionality
    avg_pool3 = AvgPool2D(pool_size=(2, 2), strides=2, padding='valid')(conv5)

    # Flatten the feature maps
    flatten_layer = Flatten()(avg_pool3)

    # Two fully connected layers with dropout
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout1 = Dropout(rate=0.2)(dense1)
    dense2 = Dense(units=64, activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.2)(dense2)
    output_layer = Dense(units=1000, activation='softmax')(dropout2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model