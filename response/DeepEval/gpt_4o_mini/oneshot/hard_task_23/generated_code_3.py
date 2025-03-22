import keras
from keras.layers import Input, Conv2D, AveragePooling2D, UpSampling2D, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Initial 1x1 convolutional layer
    initial_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # First branch: Local feature extraction with two sequential 3x3 convolutional layers
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(initial_conv)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch1)

    # Second branch: Average pooling followed by a 3x3 convolution and then upsampling
    branch2 = AveragePooling2D(pool_size=(2, 2))(initial_conv)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)

    # Third branch: Average pooling followed by a 3x3 convolution and then upsampling
    branch3 = AveragePooling2D(pool_size=(2, 2))(initial_conv)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)

    # Concatenate the outputs of all branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Convolutional layer to refine the concatenated output
    refined_output = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(concatenated)

    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(refined_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model