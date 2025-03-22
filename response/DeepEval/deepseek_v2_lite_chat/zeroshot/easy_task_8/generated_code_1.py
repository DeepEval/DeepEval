import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, DepthwiseConv2D, AveragePooling2D, Concatenate, Flatten, Dense, Dropout
from keras.layers import SeparableConv2D
from keras.optimizers import Adam

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    input_shape = (28, 28, 1)

    # Input layer
    input_img = Input(shape=input_shape)

    # Depthwise separable convolutional layer
    depthwise = DepthwiseConv2D(kernel_size=(3, 3), depth_multiplier=1, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(input_img)
    batchnorm = AveragePooling2D(pool_size=(2, 2))(depthwise)

    # 1x1 convolutional layer for feature extraction
    one_by_one = Conv2D(64, (1, 1), activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(batchnorm)

    # Concatenate the depthwise and 1x1 conv outputs
    concatenated = Concatenate()([depthwise, one_by_one])

    # Add dropout layers to mitigate overfitting
    dropout1 = Dropout(0.5)(concatenated)
    dropout2 = Dropout(0.5)(dropout1)

    # Flatten the output
    flattened = Flatten()(dropout2)

    # Fully connected layer
    output = Dense(10, activation='softmax')(flattened)

    # Model
    model = Model(inputs=input_img, outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Get the model
model = dl_model()