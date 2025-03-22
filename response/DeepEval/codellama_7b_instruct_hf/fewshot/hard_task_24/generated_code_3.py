import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # First branch: local features through 3x3 convolutional layer
    branch1 = Conv2D(32, (3, 3), activation='relu')(input_layer)

    # Second branch: downsampling through max pooling layer
    branch2 = MaxPooling2D(pool_size=(2, 2))(input_layer)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)

    # Third branch: downsampling through max pooling layer
    branch3 = MaxPooling2D(pool_size=(2, 2))(input_layer)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)

    # Fuse the outputs of all branches
    concatenate = Concatenate()([branch1, branch2, branch3])

    # Final convolutional layer
    x = Conv2D(64, (3, 3), activation='relu')(concatenate)

    # Flatten and pass through fully connected layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=x)

    return model