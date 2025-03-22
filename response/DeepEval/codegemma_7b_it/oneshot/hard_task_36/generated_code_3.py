import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, GlobalAveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main pathway
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)
    dropout = Dropout(0.5)(max_pooling)

    # Branch pathway
    conv_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(dropout)

    # Fusion layer
    fusion = keras.layers.Add()([dropout, conv_branch])

    # Global average pooling
    gap = GlobalAveragePooling2D()(fusion)

    # Flatten layer
    flatten_layer = Flatten()(gap)

    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model creation
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model