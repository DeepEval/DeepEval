from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Flatten, Dense
from keras.models import Model


def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the 3x3 convolutional branch
    conv_3x3 = Conv2D(32, (3, 3), activation='relu')(input_shape)

    # Define the 1x1 convolutional branch
    conv_1x1 = Conv2D(32, (1, 1), activation='relu')(input_shape)
    conv_1x1 = Conv2D(32, (3, 3), activation='relu')(conv_1x1)

    # Define the max pooling branch
    max_pool = MaxPooling2D((2, 2))(input_shape)

    # Define the multi-scale feature fusion block
    fused_features = concatenate([conv_3x3, conv_1x1, max_pool])

    # Define the flattening layer
    flattened_features = Flatten()(fused_features)

    # Define the fully connected layers
    fc1 = Dense(128, activation='relu')(flattened_features)
    fc2 = Dense(10, activation='softmax')(fc1)

    # Create the model
    model = Model(input_shape, fc2)

    return model