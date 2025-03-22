import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dropout, Dense, Concatenate, Reshape, Conv2D
from keras.models import Model

def dl_model():
    # Input layer
    input_tensor = Input(shape=(28, 28, 1))  # MNIST images are 28x28 pixels and grayscale

    # Block 1
    # Path 1: Average Pooling (1x1)
    path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_tensor)
    path1 = Flatten()(path1)
    path1 = Dropout(0.5)(path1)

    # Path 2: Average Pooling (2x2)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_tensor)
    path2 = Flatten()(path2)
    path2 = Dropout(0.5)(path2)

    # Path 3: Average Pooling (4x4)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_tensor)
    path3 = Flatten()(path3)
    path3 = Dropout(0.5)(path3)

    # Concatenate outputs of all paths
    block1_output = Concatenate()([path1, path2, path3])

    # Fully connected layer and reshape
    fc1 = Dense(128, activation='relu')(block1_output)
    reshaped_output = Reshape((8, 8, 2))(fc1)  # Reshape to 4D tensor (example dimensions)

    # Block 2
    # Branch 1: 1x1 Convolution
    branch1 = Conv2D(32, kernel_size=(1, 1), activation='relu')(reshaped_output)

    # Branch 2: 3x3 Convolution
    branch2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(reshaped_output)

    # Branch 3: (1x1 Convolution followed by 3x3 Convolution)
    branch3 = Conv2D(32, kernel_size=(1, 1), activation='relu')(reshaped_output)
    branch3 = Conv2D(32, kernel_size=(3, 3), activation='relu')(branch3)

    # Branch 4: (Average Pooling followed by 1x1 Convolution)
    branch4 = AveragePooling2D(pool_size=(2, 2))(reshaped_output)
    branch4 = Conv2D(32, kernel_size=(1, 1), activation='relu')(branch4)

    # Concatenate outputs of all branches
    block2_output = Concatenate()([branch1, branch2, branch3, branch4])

    # Fully connected layers for classification
    block2_output_flat = Flatten()(block2_output)
    fc2 = Dense(128, activation='relu')(block2_output_flat)
    output = Dense(10, activation='softmax')(fc2)  # 10 classes for MNIST

    # Create the model
    model = Model(inputs=input_tensor, outputs=output)
    
    return model

# Example of creating the model
model = dl_model()
model.summary()