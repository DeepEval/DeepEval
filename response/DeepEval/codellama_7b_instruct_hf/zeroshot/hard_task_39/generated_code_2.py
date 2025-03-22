import keras
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: 3 max pooling layers with varying scales
    x = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(x)
    x = Flatten()(x)

    # Block 2: Multiple branches
    branch1 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(64, (3, 3), activation='relu')(input_layer)
    branch3 = Conv2D(64, (5, 5), activation='relu')(input_layer)
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='valid')(input_layer)
    x = keras.layers.concatenate([branch1, branch2, branch3, branch4])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=x)
    return model