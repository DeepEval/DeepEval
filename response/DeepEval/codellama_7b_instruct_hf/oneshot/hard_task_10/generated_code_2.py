import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first feature extraction path
    path1 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    path1 = MaxPooling2D((2, 2))(path1)

    # Define the second feature extraction path
    path2 = Conv2D(64, (1, 7), activation='relu')(input_layer)
    path2 = Conv2D(64, (7, 1), activation='relu')(path2)
    path2 = MaxPooling2D((2, 2))(path2)

    # Concatenate the outputs of the two paths
    merged_path = Concatenate()([path1, path2])

    # Apply a 1x1 convolution to align the output dimensions with the input image's channel
    merged_path = Conv2D(32, (1, 1), activation='relu')(merged_path)

    # Merge the main path and the branch through addition
    merged_path = keras.layers.add([merged_path, input_layer])

    # Flatten the output of the merged path
    merged_path = Flatten()(merged_path)

    # Apply a fully connected layer and a softmax activation function
    merged_path = Dense(10, activation='softmax')(merged_path)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=merged_path)

    return model