from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def depthwise_conv2d(input_tensor, kernel_size, padding='same', strides=(1, 1)):
    return layers.DepthwiseConv2D(
        kernel_size, padding=padding, strides=strides, use_bias=False)(input_tensor)

def pointwise_conv2d(input_tensor, filters, kernel_size=(1, 1), padding='same', strides=(1, 1)):
    return layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, use_bias=False)(input_tensor)

def residual_module(input_tensor):
    # Split the input into three groups
    x1 = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)

    # Depthwise separable convolutional layers
    x1 = layers.Lambda(lambda x: pointwise_conv2d(depthwise_conv2d(x, (1, 1)), 32, (1, 1)))(x1[0])
    x2 = layers.Lambda(lambda x: pointwise_conv2d(depthwise_conv2d(x, (3, 3)), 32, (1, 1)))(x1[1])
    x3 = layers.Lambda(lambda x: pointwise_conv2d(depthwise_conv2d(x, (5, 5)), 32, (1, 1)))(x1[2])

    # Concatenate the outputs
    x = layers.concatenate([x1, x2, x3])

    # Batch normalization
    x = layers.BatchNormalization()(x)

    # Return the residual output
    return x

def branch_module(input_tensor):
    # 1x1 convolution branch
    x1 = layers.Conv2D(16, (1, 1), padding='same')(input_tensor)

    # 3x3 convolution branch
    x2 = layers.Conv2D(16, (3, 3), padding='same')(input_tensor)

    # 1x7 convolution, 7x1 convolution, 3x3 convolution branch
    x3 = layers.Conv2D(16, (1, 7), padding='same')(input_tensor)
    x3 = layers.Conv2D(16, (7, 1), padding='same')(x3)
    x3 = layers.Conv2D(16, (3, 3), padding='same')(x3)

    # Average pooling branch
    x4 = layers.AveragePooling2D()(input_tensor)

    # Concatenate the outputs
    x = layers.concatenate([x1, x2, x3, x4])

    # Batch normalization
    x = layers.BatchNormalization()(x)

    # Return the branch output
    return x

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize image data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert labels to one-hot encoded vectors
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Input layer
    input_tensor = layers.Input(shape=(32, 32, 3))

    # First block
    x = residual_module(input_tensor)

    # Second block
    x = branch_module(x)

    # Classification layers
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)

    # Model definition
    model = models.Model(inputs=input_tensor, outputs=output_tensor)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32)

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print('Test accuracy:', accuracy)

    return model