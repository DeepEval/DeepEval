from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Flatten, multiply
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the 1x1 convolution layer
    conv1x1 = Conv2D(filters=3, kernel_size=1, strides=1, activation='relu')

    # Define the 3x3 depthwise separable convolution layer
    conv3x3 = Conv2D(filters=3, kernel_size=3, strides=1, activation='relu', use_bias=False, depthwise_initializer='he_normal')

    # Define the global average pooling layer
    pool = GlobalAveragePooling2D()

    # Define the fully connected layers
    fc1 = Dense(128, activation='relu')
    fc2 = Dense(10, activation='softmax')

    # Define the model
    model = Model(inputs=Input(shape=input_shape), outputs=conv1x1(conv3x3(pool(conv3x3(conv1x1(pool(input_shape)))))))

    # Add the channel attention weights
    model.add(Flatten())
    model.add(fc1)
    model.add(fc2)

    return model