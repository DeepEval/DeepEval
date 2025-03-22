import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Path 1: Single 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)

    # Path 2: Average pooling followed by a 1x1 convolution
    path2 = MaxPooling2D(pool_size=(2, 2))(path1)
    path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(path2)

    # Path 3: 1x1 convolution followed by two parallel 1x3 and 3x1 convolutions
    path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)
    path3 = Conv2D(filters=32, kernel_size=(3, 1), activation='relu')(path3)
    path3 = Conv2D(filters=32, kernel_size=(1, 3), activation='relu')(path3)

    # Path 4: 1x1 convolution followed by a 3x3 convolution, then two parallel 1x3 and 3x1 convolutions
    path4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)
    path4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(path4)
    path4 = Conv2D(filters=32, kernel_size=(1, 3), activation='relu')(path4)
    path4 = Conv2D(filters=32, kernel_size=(3, 1), activation='relu')(path4)

    # Concatenate the outputs of all paths
    concat = Add()([path1, path2, path3, path4])

    # Fully connected layer for classification
    outputs = Flatten()(concat)
    outputs = Dense(units=10, activation='softmax')(outputs)

    # Model
    model = Model(inputs=inputs, outputs=outputs)

    return model