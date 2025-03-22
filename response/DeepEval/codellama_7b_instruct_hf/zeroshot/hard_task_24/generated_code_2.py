from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first 1x1 convolutional layer
    conv1 = Conv2D(32, (1, 1), padding='same')(input_shape)

    # Define the first branch
    branch1 = Conv2D(64, (3, 3), padding='same')(conv1)

    # Define the second branch
    branch2 = MaxPooling2D((2, 2))(conv1)
    branch2 = Conv2D(64, (3, 3), padding='same')(branch2)
    branch2 = UpSampling2D((2, 2))(branch2)

    # Define the third branch
    branch3 = MaxPooling2D((2, 2))(conv1)
    branch3 = Conv2D(64, (3, 3), padding='same')(branch3)
    branch3 = UpSampling2D((2, 2))(branch3)

    # Define the fusion layer
    fusion = Concatenate()([branch1, branch2, branch3])

    # Define the final 1x1 convolutional layer
    final_conv = Conv2D(128, (1, 1), padding='same')(fusion)

    # Define the fully connected layers
    x = Flatten()(final_conv)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(input_shape, x)

    return model