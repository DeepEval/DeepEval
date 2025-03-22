import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model
from keras.regularizers import L1L2

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution, 3x3 convolution
    branch1 = Conv2D(32, (1, 1), activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01))(input_layer)
    branch1 = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01))(branch1)

    # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution
    branch2 = Conv2D(32, (1, 1), activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01))(input_layer)
    branch2 = Conv2D(32, (1, 7), activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01))(branch2)
    branch2 = Conv2D(32, (7, 1), activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01))(branch2)
    branch2 = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01))(branch2)

    # Branch 3: max pooling
    branch3 = MaxPooling2D((2, 2))(input_layer)

    # Concat branch outputs
    output = Concatenate()([branch1, branch2, branch3])

    # Batch normalization
    output = BatchNormalization()(output)

    # Flatten
    output = Flatten()(output)

    # Dense layers
    output = Dense(128, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01))(output)
    output = Dense(64, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01))(output)
    output = Dense(10, activation='softmax')(output)

    # Create model
    model = Model(inputs=input_layer, outputs=output)

    return model