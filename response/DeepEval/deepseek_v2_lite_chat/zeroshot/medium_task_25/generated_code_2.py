import keras
from keras.models import Model
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, ZeroPadding2D, MaxPooling2D, Flatten, Dense
from keras.layers import BatchNormalization

def dl_model():
    input_shape = (32, 32, 3)  # Input image dimensions (CIFAR-10 standard)
    num_classes = 10  # Number of classes in CIFAR-10

    # Path 1: Single 1x1 convolution
    path1_input = Input(shape=input_shape)
    path1_conv = Conv2D(32, (1, 1), activation='relu')(path1_input)

    # Path 2: Average pooling followed by a 1x1 convolution
    path2_input = AveragePooling2D(pool_size=(2, 2))(path1_conv)
    path2_conv = Conv2D(32, (1, 1), activation='relu')(path2_input)

    # Path 3: 1x1 convolution followed by two parallel convolutions
    path3_input = Conv2D(32, (1, 1), activation='relu')(path1_conv)
    path3_conv1 = Conv2D(32, (1, 1), activation='relu')(path3_input)
    path3_conv2 = Conv2D(32, (1x3, 3x1), activation='relu', padding='same')(path3_input)

    # Path 4: 1x1 convolution followed by a 3x3 convolution, then two parallel convolutions
    path4_input = Conv2D(32, (1, 1), activation='relu')(path1_conv)
    path4_conv1 = Conv2D(32, (3, 3), activation='relu')(path4_input)
    path4_conv2 = Conv2D(32, (1x3, 3x1), activation='relu', padding='same')(path4_input)

    # Concatenate and flatten the features from all paths
    concat = Concatenate()([path2_conv, path3_conv1, path3_conv2, path4_conv1])
    flatten = Flatten()(concat)

    # Fully connected layer for classification
    output = Dense(num_classes, activation='softmax')(flatten)

    # Create the model
    model = Model(inputs=[path1_input, path2_input, path3_input, path4_input], outputs=output)

    return model

# Instantiate and return the model
model = dl_model()
model.summary()