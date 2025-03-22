from keras.layers import Input, Conv2D, MaxPooling2D, AvgPool2D, Flatten, Concatenate, Dense, Dropout, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    # 3 parallel paths, each with different average pooling layers
    path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    output1 = Concatenate()([path1, path2, path3])

    # Fully connected layer between block 1 and block 2
    output1 = Dense(units=128, activation='relu')(output1)

    # Block 2
    # 3 branches, each with different convolutional and pooling layers
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output1)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(output1)
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(output1)
    branch4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output1)
    branch5 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(output1)
    branch6 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(output1)
    branch7 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(output1)
    branch8 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(output1)
    output2 = Concatenate()([branch1, branch2, branch3, branch4, branch5, branch6, branch7, branch8])

    # Flatten and dense layers
    output2 = Flatten()(output2)
    output2 = Dense(units=128, activation='relu')(output2)
    output2 = Dense(units=10, activation='softmax')(output2)

    model = Model(inputs=input_layer, outputs=output2)

    return model