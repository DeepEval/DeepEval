import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the last dimension
    groups = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    # Define the first block
    conv1 = Conv2D(32, (1, 1), activation='relu', name='conv1')(groups[0])
    conv2 = Conv2D(32, (3, 3), activation='relu', name='conv2')(groups[1])
    conv3 = Conv2D(32, (5, 5), activation='relu', name='conv3')(groups[2])
    concatenated = Concatenate()([conv1, conv2, conv3])

    # Define the second block
    branch1 = Conv2D(64, (1, 1), activation='relu', name='conv1_1')(input_layer)
    branch2 = Conv2D(64, (3, 3), activation='relu', name='conv1_3')(input_layer)
    branch3 = Conv2D(64, (5, 5), activation='relu', name='conv1_5')(input_layer)
    branch4 = MaxPooling2D((1, 1), name='maxpool')(input_layer)
    concatenated_2 = Concatenate()([branch1, branch2, branch3, branch4])

    # Define the final layers
    batch_norm = BatchNormalization()(concatenated_2)
    flatten = Flatten()(batch_norm)
    dense = Dense(10, activation='softmax', name='dense')(flatten)

    # Create the model
    model = Model(inputs=input_layer, outputs=dense)

    return model