import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Branch 1: 3x3 Convolutions
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Branch 2: 1x1 Convolutions followed by two 3x3 Convolutions
    branch2 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch2)

    # Branch 3: Max Pooling
    branch3 = MaxPooling2D(pool_size=(2, 2), padding='same')(input_layer)

    # Concatenate the outputs of the branches
    merged = Concatenate()([branch1, branch2, branch3])

    # Flatten the result
    flatten_layer = Flatten()(merged)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Output layer with softmax activation for classification into 10 classes
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to create the model
model = dl_model()
model.summary()  # To view the model's architecture