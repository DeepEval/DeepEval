import keras
from keras.datasets import mnist
from keras.layers import Input, Conv2D, DepthwiseConv2D, Conv2DTranspose, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape data to include channel dimension (28x28 images)
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    # Normalize the pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Define the input shape
    input_shape = (28, 28, 1)

    # Encoder layers
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (1, 1), activation='relu')(input_layer)  # 1x1 convolution to increase dimensionality
    x = DepthwiseConv2D((3, 3), activation='relu')(x)  # Depthwise separable convolution for feature extraction
    x = Conv2D(64, (1, 1), activation='relu')(x)  # 1x1 convolution to reduce dimensionality
    x = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x, strides=(2, 2))  # Stride of 2

    # Decoder layers
    decoder_input = Conv2DTranspose(64, (1, 1), activation='relu')(x)
    decoder_output = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same')(decoder_input)

    # Combine encoder and decoder
    model = Model(inputs=input_layer, outputs=decoder_output)

    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Build the model
model = dl_model()