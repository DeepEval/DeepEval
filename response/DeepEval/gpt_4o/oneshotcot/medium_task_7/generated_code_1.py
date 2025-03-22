from keras.layers import Input, Conv2D, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First path: Sequential convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv2)

    # Second path: Direct convolution of input
    direct_conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)

    # Add the outputs from conv1, conv2, conv3 and direct_conv
    combined = Add()([conv3, direct_conv])

    # Flatten and add dense layers for classification
    flatten_layer = Flatten()(combined)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model