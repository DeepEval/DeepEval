from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import LeakyReLU, Activation

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the inputs
    inputs = Input(shape=input_shape, name='input')

    # Define the branch 1: 1x1 convolution
    branch1 = Conv2D(32, (1, 1), activation='relu', name='branch1_conv1')(inputs)
    branch1 = Dropout(0.2, name='branch1_dropout1')(branch1)

    # Define the branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(32, (1, 1), activation='relu', name='branch2_conv1')(inputs)
    branch2 = Conv2D(32, (3, 3), activation='relu', name='branch2_conv2')(branch2)
    branch2 = Dropout(0.2, name='branch2_dropout')(branch2)

    # Define the branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    branch3 = Conv2D(32, (1, 1), activation='relu', name='branch3_conv1')(inputs)
    branch3 = Conv2D(32, (3, 3), activation='relu', name='branch3_conv2')(branch3)
    branch3 = Conv2D(32, (3, 3), activation='relu', name='branch3_conv3')(branch3)
    branch3 = Dropout(0.2, name='branch3_dropout')(branch3)

    # Define the branch 4: Average pooling followed by 1x1 convolution
    branch4 = AveragePooling2D((2, 2))(inputs)
    branch4 = Conv2D(32, (1, 1), activation='relu', name='branch4_conv')(branch4)
    branch4 = Dropout(0.2, name='branch4_dropout')(branch4)

    # Concatenate the outputs from all branches
    concatenated = Concatenate()([branch1, branch2, branch3, branch4])

    # Define the fully connected layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(concatenated)
    x = Dropout(0.2)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(10, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)

    # Define the final output layer
    outputs = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation='softmax')(outputs)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs, name='image_classification_model')

    return model