import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Branch 1: 1x1 convolution followed by 3x3 convolution
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch1)
    branch1 = Dropout(0.3)(branch1)  # Dropout layer to mitigate overfitting

    # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, then 3x3 convolution
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
    branch2 = Dropout(0.3)(branch2)  # Dropout layer

    # Branch 3: Max pooling
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    branch3 = Dropout(0.3)(branch3)  # Dropout layer

    # Concatenate outputs from all branches
    concatenated_branches = Concatenate()([branch1, branch2, branch3])

    # Flatten and pass through fully connected layers
    flatten_layer = Flatten()(concatenated_branches)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model