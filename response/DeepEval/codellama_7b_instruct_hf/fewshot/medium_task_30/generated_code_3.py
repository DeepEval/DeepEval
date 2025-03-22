from keras.models import Model
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, AveragePooling2D
from keras.applications import VGG16

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first average pooling layer
    avgpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    flatten1 = Flatten()(avgpool1)

    # Define the second average pooling layer
    avgpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    flatten2 = Flatten()(avgpool2)

    # Define the third average pooling layer
    avgpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    flatten3 = Flatten()(avgpool3)

    # Concatenate the outputs of the three average pooling layers
    concat_layer = Concatenate()([flatten1, flatten2, flatten3])

    # Flatten the concatenated output
    flatten_layer = Flatten()(concat_layer)

    # Add two fully connected layers
    dense_layer1 = Dense(128, activation='relu')(flatten_layer)
    dense_layer2 = Dense(64, activation='relu')(dense_layer1)

    # Define the output layer
    output_layer = Dense(10, activation='softmax')(dense_layer2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model