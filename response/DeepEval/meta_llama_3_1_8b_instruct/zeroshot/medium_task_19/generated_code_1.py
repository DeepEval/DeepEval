from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = keras.Input(shape=input_shape)

    # First branch: Dimensionality reduction using 1x1 convolution
    branch1 = layers.Conv2D(32, 1, activation='relu', name='branch1_conv1')(inputs)
    branch1 = layers.BatchNormalization(name='branch1_bn')(branch1)

    # Second branch: Feature extraction using 1x1 and 3x3 convolution
    branch2 = layers.Conv2D(32, 1, activation='relu', name='branch2_conv1')(inputs)
    branch2 = layers.BatchNormalization(name='branch2_bn1')(branch2)
    branch2 = layers.Conv2D(32, 3, activation='relu', name='branch2_conv2')(branch2)
    branch2 = layers.BatchNormalization(name='branch2_bn2')(branch2)

    # Third branch: Capturing larger spatial information using 1x1 and 5x5 convolution
    branch3 = layers.Conv2D(32, 1, activation='relu', name='branch3_conv1')(inputs)
    branch3 = layers.BatchNormalization(name='branch3_bn1')(branch3)
    branch3 = layers.Conv2D(32, 5, activation='relu', name='branch3_conv2')(branch3)
    branch3 = layers.BatchNormalization(name='branch3_bn2')(branch3)

    # Fourth branch: Downsampling and further processing using max pooling and 1x1 convolution
    branch4 = layers.MaxPooling2D((3, 3), strides=2, name='branch4_maxpool')(inputs)
    branch4 = layers.Conv2D(32, 1, activation='relu', name='branch4_conv1')(branch4)
    branch4 = layers.BatchNormalization(name='branch4_bn')(branch4)

    # Concatenate the outputs of the four branches
    concatenated = layers.Concatenate()([branch1, branch2, branch3, branch4])

    # Apply a global average pooling layer to reduce spatial dimensions
    gap = layers.GlobalAveragePooling2D(name='gap')(concatenated)

    # Add two fully connected layers for classification
    x = layers.Dense(128, activation='relu', name='dense1')(gap)
    outputs = layers.Dense(10, activation='softmax', name='dense2')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model