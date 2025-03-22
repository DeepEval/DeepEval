from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, LayerNormalization, MaxPooling2D, Concatenate, Dense, Add
from keras.layers import BatchNormalization, Activation

def dl_model():
    # Parameters
    img_rows, img_cols, img_channels = 32, 32, 3
    img_input = Input(shape=(img_rows, img_cols, img_channels))
    num_classes = 10

    # First convolutional layer with depthwise separable convolution
    x = Conv2D(32, (7, 7), strides=(2, 2), padding="same", use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(32, (1, 1), use_bias=False)(x)
    x = LayerNormalization(epsilon=1e-5)(x)

    # MaxPooling
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Second convolutional layer
    x = Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Fully connected layer with input channels equal to output channels of second conv
    fc1 = Dense(64, use_bias=False)(x)
    fc1 = BatchNormalization()(fc1)
    fc1 = Activation("relu")(fc1)

    # Fully connected layer with input channels equal to output channels of first conv
    fc2 = Dense(64, use_bias=False)(fc1)
    fc2 = BatchNormalization()(fc2)
    fc2 = Activation("relu")(fc2)

    # Output layer
    combined = Add()([img_input, fc2])
    output1 = Dense(512, activation="relu")(combined)
    output2 = Dense(num_classes, activation="softmax")(output1)

    # Model
    model = Model(inputs=img_input, outputs=output2)
    return model

# Instantiate the model
model = dl_model()
model.summary()