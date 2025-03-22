import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(224, 224, 3))

    # Add two sequential feature extraction layers
    conv_layer1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool_layer1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_layer1)

    conv_layer2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool_layer1)
    pool_layer2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_layer2)

    # Add three additional convolutional layers
    conv_layer3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool_layer2)
    conv_layer4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_layer3)
    conv_layer5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_layer4)

    # Add another average pooling layer to reduce the dimensionality of the feature maps
    pool_layer3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_layer5)

    # Flatten the feature maps
    flatten_layer = Flatten()(pool_layer3)

    # Add two fully connected layers with dropout
    dense_layer1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout_layer1 = Dropout(rate=0.5)(dense_layer1)
    dense_layer2 = Dense(units=64, activation='relu')(dropout_layer1)
    dropout_layer2 = Dropout(rate=0.5)(dense_layer2)

    # Add a softmax output layer with 1,000 neurons
    output_layer = Dense(units=1000, activation='softmax')(dropout_layer2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model