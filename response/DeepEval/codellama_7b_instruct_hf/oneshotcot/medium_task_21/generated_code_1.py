from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # branch 1: 1x1 convolution
    branch1 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch1 = Dropout(0.2)(branch1)

    # branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)
    branch2 = Dropout(0.2)(branch2)

    # branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    branch3 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)
    branch3 = Dropout(0.2)(branch3)

    # branch 4: average pooling followed by 1x1 convolution
    branch4 = MaxPooling2D(pool_size=(2, 2))(input_layer)
    branch4 = Conv2D(32, (1, 1), activation='relu')(branch4)
    branch4 = Dropout(0.2)(branch4)

    # concatenate outputs from all branches
    concatenated = Concatenate()([branch1, branch2, branch3, branch4])

    # apply batch normalization and flatten the result
    flattened = BatchNormalization()(concatenated)
    flattened = Flatten()(flattened)

    # add three fully connected layers for classification
    output = Dense(10, activation='softmax')(flattened)

    # create and return the model
    model = Model(inputs=input_layer, outputs=output)
    return model