import keras
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Conv2DTranspose, Input, Flatten, Dense
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import to_categorical


def dl_model():
    
    # Load and preprocess the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Define the input shape
    input_shape = (32, 32, 3)  # This matches the expected input shape of CIFAR-10
    input_tensor = Input(shape=input_shape)

    # First branch: 1x1 convolution, then sequential 3x3 convolutions
    x = Conv2D(64, (1, 1), activation='relu')(input_tensor)
    x = Conv2D(64, (3, 3), activation='relu')(x)

    # Second branch: Average Pooling, then 3x3 convolution, transposed convolution for upsampling
    y = MaxPooling2D((3, 3), strides=(3, 3))(input_tensor)
    y = Conv2D(64, (3, 3), activation='relu')(y)
    y = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu')(y)  # Upsampling by a factor of 2

    # Third branch: Average Pooling, then 3x3 convolution, transposed convolution for upsampling
    z = MaxPooling2D((3, 3), strides=(3, 3))(input_tensor)
    z = Conv2D(64, (3, 3), activation='relu')(z)
    z = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu')(z)  # Upsampling by a factor of 2

    # Concatenate the outputs of the three branches
    concat = Concatenate()([x, y, z])

    # Flatten and feed into a fully connected layer
    flatten = Flatten()(concat)
    output = Dense(10, activation='softmax')(flatten)

    # Define the model
    model = Model(inputs=input_tensor, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print model summary
    model.summary()

    return model

# Call the function to create and compile the model
model = dl_model()