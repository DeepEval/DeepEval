from keras.models import Model
from keras.layers import Input, Dense, Flatten, Concatenate, AveragePooling2D, Conv2D, Reshape, MaxPooling2D

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Block 1
    # Three parallel paths
    path1 = AveragePooling2D(pool_size=(1, 1), strides=None, padding='valid')(input_shape)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')(input_shape)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=None, padding='valid')(input_shape)

    # Flatten and concatenate the paths
    x = Flatten()(Concatenate()([path1, path2, path3]))

    # Fully connected layer and reshape
    x = Dense(64, activation='relu')(x)
    x = Reshape((4, 4, 16))(x)

    # Block 2
    # Three branches
    branch1 = Conv2D(16, (1, 1), activation='relu')(x)
    branch2 = Conv2D(32, (3, 3), activation='relu')(x)
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=None, padding='valid')(x)
    branch4 = Conv2D(64, (3, 3), activation='relu')(x)

    # Concatenate the branches
    x = Concatenate()([branch1, branch2, branch3, branch4])

    # Fully connected layers and output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_shape, outputs=x)

    return model