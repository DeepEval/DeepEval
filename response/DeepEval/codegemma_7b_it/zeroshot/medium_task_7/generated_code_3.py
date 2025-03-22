from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Add
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Path 1: Sequential convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=3, activation='relu')(inputs)
    conv2 = Conv2D(filters=64, kernel_size=3, activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=3, activation='relu')(conv2)

    # Path 2: Separate convolutional layer
    conv4 = Conv2D(filters=64, kernel_size=3, activation='relu')(inputs)

    # Add outputs from both paths
    concat = Add()([conv3, conv4])

    # Flatten and fully connected layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=512, activation='relu')(flatten)
    outputs = Dense(units=10, activation='softmax')(dense1)

    # Model construction
    model = Model(inputs=inputs, outputs=outputs)

    return model