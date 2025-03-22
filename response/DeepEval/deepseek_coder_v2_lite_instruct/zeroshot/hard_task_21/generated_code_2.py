import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, Add, Lambda, Concatenate, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Define the input layer
    inputs = Input(shape=(32, 32, 3))

    # Main path
    main_path = Conv2D(32, (1, 1), activation='relu')(inputs)
    main_path = DepthwiseConv2D(kernel_size=(3, 3), activation='relu')(main_path)
    main_path = Conv2D(32, (1, 1), activation='relu')(main_path)
    main_path = DepthwiseConv2D(kernel_size=(5, 5), activation='relu')(main_path)
    main_path = Conv2D(32, (1, 1), activation='relu')(main_path)

    # Split the main path into three groups
    main_path_1x1 = Lambda(lambda x: x[:, :, :, :32])(main_path)
    main_path_3x3 = Lambda(lambda x: x[:, :, :, 32:64])(main_path)
    main_path_5x5 = Lambda(lambda x: x[:, :, :, 64:])(main_path)

    # Feature extraction for each group
    main_path_1x1 = Conv2D(32, (1, 1), activation='relu')(main_path_1x1)
    main_path_3x3 = Conv2D(32, (1, 1), activation='relu')(main_path_3x3)
    main_path_5x5 = Conv2D(32, (1, 1), activation='relu')(main_path_5x5)

    # Concatenate the outputs from the three groups
    main_output = Concatenate()([main_path_1x1, main_path_3x3, main_path_5x5])

    # Branch path
    branch_path = Conv2D(32, (1, 1), activation='relu')(inputs)

    # Add the main and branch paths
    added_output = Add()([main_output, branch_path])

    # Flatten and add fully connected layers
    flattened = Flatten()(added_output)
    outputs = Dense(10, activation='softmax')(flattened)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage
model = dl_model()
model.summary()