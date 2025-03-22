from keras.models import Model
from keras.layers import Input, Flatten, Dense, Conv2D, DepthwiseSeparableConv2D, Add
from keras.applications.vgg16 import VGG16

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu')(input_layer)
    x = DepthwiseSeparableConv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Add()([input_layer, x])

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = DepthwiseSeparableConv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Add()([input_layer, x])

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = DepthwiseSeparableConv2D(256, (3, 3), activation='relu')(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = Add()([input_layer, x])

    # Flatten layer
    x = Flatten()(x)

    # Output layer
    x = Dense(10, activation='softmax')(x)

    # Create model
    model = Model(inputs=input_layer, outputs=x)

    return model