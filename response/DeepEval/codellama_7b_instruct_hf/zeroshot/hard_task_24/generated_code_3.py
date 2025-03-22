import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # 1x1 initial convolutional layer
    x = Conv2D(32, (1, 1), activation='relu')(input_layer)

    # Branch 1
    branch1 = Conv2D(32, (3, 3), activation='relu')(x)

    # Branch 2
    branch2 = MaxPooling2D((2, 2))(x)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)

    # Branch 3
    branch3 = MaxPooling2D((2, 2))(x)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)

    # Fusion layer
    x = concatenate([branch1, branch2, branch3])

    # 1x1 convolutional layer
    x = Conv2D(32, (1, 1), activation='relu')(x)

    # Flatten
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    # Output layer
    output_layer = Dense(10, activation='softmax')(x)

    # Create and return the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model