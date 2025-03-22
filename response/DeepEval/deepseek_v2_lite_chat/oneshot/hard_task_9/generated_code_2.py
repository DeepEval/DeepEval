import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the RGB codes to the range [0, 1]
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Define the input tensor
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv2)

    # Branch 3: 1x1 convolution followed by two 3x3 convolutions
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    conv5 = Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu')(conv4)
    conv6 = Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu')(conv5)

    # Concatenate the outputs from the three branches
    concat = Concatenate(axis=-1)([conv1, conv3, conv6])

    # Adjust the output dimensions to match the input image's channel size
    adjust = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(concat)

    # Main path: branch directly connects to input, and main path and branch are fused together through addition
    fused = keras.layers.Add()([input_layer, adjust])

    # Add batch normalization and flatten layers
    batch_norm = BatchNormalization()(fused)
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers for classification
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()

# Display the model summary
model.summary()