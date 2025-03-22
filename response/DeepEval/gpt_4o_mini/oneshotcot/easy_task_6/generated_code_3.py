import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))

    # Step 2: Main Path
    # First convolutional layer (increasing feature map width)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    # Second convolutional layer (restoring number of channels)
    conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Step 3: Branch Path
    branch_path = input_layer  # Directly connecting to the input

    # Step 4: Combine paths
    combined = Add()([conv2, branch_path])  # Element-wise addition

    # Step 5: Flatten layer
    flatten_layer = Flatten()(combined)

    # Step 6: Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for MNIST

    # Building the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model