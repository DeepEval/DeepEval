import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first sequential block
    conv_layer1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pooling_layer1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_layer1)

    # Define the second sequential block
    conv_layer2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling_layer1)
    max_pooling_layer2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_layer2)

    # Define the third sequential block
    conv_layer3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling_layer2)
    max_pooling_layer3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_layer3)

    # Flatten the feature maps
    flatten_layer = Flatten()(max_pooling_layer3)

    # Define the fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model