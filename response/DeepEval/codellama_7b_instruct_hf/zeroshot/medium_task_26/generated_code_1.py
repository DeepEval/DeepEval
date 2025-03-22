from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 64)

    # Define the first convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=3, activation='relu')(input_shape)
    # Define the second convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=3, activation='relu')(conv1)
    # Define the third convolutional layer
    conv3 = Conv2D(filters=32, kernel_size=3, activation='relu')(conv2)
    # Define the fourth convolutional layer
    conv4 = Conv2D(filters=32, kernel_size=3, activation='relu')(conv3)
    # Define the pooling layer
    pool = MaxPooling2D(pool_size=2)(conv4)
    # Define the flatten layer
    flat = Flatten()(pool)
    # Define the first fully connected layer
    fc1 = Dense(units=128, activation='relu')(flat)
    # Define the second fully connected layer
    fc2 = Dense(units=64, activation='relu')(fc1)
    # Define the output layer
    output = Dense(units=10, activation='softmax')(fc2)

    # Create the model
    model = Model(inputs=input_shape, outputs=output)

    return model