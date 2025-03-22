import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first sequential block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    # Define the second sequential block
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Define the third sequential block
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling2)
    max_pooling3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)

    # Define the fourth sequential block
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling3)
    max_pooling4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv4)

    # Flatten the feature maps
    flatten = Flatten()(max_pooling4)

    # Define the first fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten)

    # Define the second fully connected layer
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Define the output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model