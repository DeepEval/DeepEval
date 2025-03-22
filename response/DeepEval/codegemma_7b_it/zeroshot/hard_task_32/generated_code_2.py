from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Dropout, SeparableConv2D

def custom_block(x, filters, kernel_size):
    # Depthwise separable convolutional layer
    x = SeparableConv2D(filters, kernel_size, padding='same', activation='relu')(x)

    # 1x1 convolutional layer to extract features
    x = Conv2D(filters, 1, padding='same', activation='relu')(x)

    # Dropout layer after convolutional layers
    x = Dropout(0.2)(x)

    return x

def dl_model():

    # Input layer
    input_img = Input(shape=(28, 28, 1))

    # Branch 1
    branch1 = custom_block(input_img, 16, 3)

    # Branch 2
    branch2 = custom_block(input_img, 32, 5)

    # Branch 3
    branch3 = custom_block(input_img, 64, 7)

    # Concatenate outputs from all branches
    concat = concatenate([branch1, branch2, branch3])

    # Fully connected layers
    x = Flatten()(concat)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)

    # Model definition
    model = Model(inputs=input_img, outputs=x)

    return model