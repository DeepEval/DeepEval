from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, ZeroPadding2D, BatchNormalization, Concatenate, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    input_shape = (28, 28, 1)

    # 1x1 convolutional layer to increase dimensionality
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(x_train[0])
    
    # Depthwise separable 3x3 convolutional layer
    x = DepthwiseConv2D((3, 3), activation='relu', depth_multiplier=1)(x)
    
    # 1x1 convolutional layer to reduce dimensionality
    x = Conv2D(16, (1, 1), activation='relu', padding='same')(x)

    # Apply stride of 2 to all convolutional layers
    x = Conv2D(8, (2, 2), activation='relu', padding='same')(x)

    # Concatenate the outputs of the different layers
    merged = Concatenate()([x, x])

    # Flatten the concatenated output for fully connected layer
    x = Flatten()(merged)

    # Fully connected layer for classification
    output = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=[input_shape], outputs=[output])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model