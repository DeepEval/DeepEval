from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Conv2DTranspose, Flatten, Dense
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the model architecture
def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # Assuming input image size is 32x32

    # Branch 1: Local feature extraction with a 3x3 convolutional layer
    conv1 = Conv2D(64, (3, 3), activation='relu')(input_layer)

    # Branch 2: Max pooling, then a convolutional layer, and another max pooling
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Branch 3: Max pooling, then a convolutional layer, and an upsampling
    conv3 = Conv2D(64, (3, 3), activation='relu')(pool2)
    upconv3 = UpSampling2D(size=(2, 2))(conv3)
    merge1 = Concatenate()([conv3, upconv3])

    # Final layers: 1x1 convolution, flattening, and three fully connected layers
    conv4 = Conv2D(10, (1, 1), activation='softmax')(merge1)
    model = Model(inputs=input_layer, outputs=conv4)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Build and return the model
model = dl_model()