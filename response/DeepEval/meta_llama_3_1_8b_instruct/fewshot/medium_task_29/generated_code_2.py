import keras
from keras.layers import Input, MaxPooling2D, Flatten, Concatenate, Dense

def dl_model():
    # Define the input layer with shape (32, 32, 3) for CIFAR-10 dataset
    input_layer = Input(shape=(32, 32, 3))

    # Define the first max pooling layer with window size 1x1 and stride 1x1
    maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    # Flatten the output of maxpool1 into a one-dimensional vector
    flatten1 = Flatten()(maxpool1)

    # Define the second max pooling layer with window size 2x2 and stride 2x2
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    # Flatten the output of maxpool2 into a one-dimensional vector
    flatten2 = Flatten()(maxpool2)

    # Define the third max pooling layer with window size 4x4 and stride 4x4
    maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    # Flatten the output of maxpool3 into a one-dimensional vector
    flatten3 = Flatten()(maxpool3)

    # Concatenate the flattened vectors from the three max pooling layers
    output_tensor = Concatenate()([flatten1, flatten2, flatten3])

    # Define the first fully connected layer with 128 units and ReLU activation
    dense1 = Dense(units=128, activation='relu')(output_tensor)

    # Define the second fully connected layer with 10 units and softmax activation for classification
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the deep learning model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model