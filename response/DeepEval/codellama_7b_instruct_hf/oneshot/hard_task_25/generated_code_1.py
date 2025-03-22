import keras
from keras.layers import Input, Conv2D, AvgPool2D, MaxPool2D, Conv2DTranspose, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(conv2)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation='relu')(conv3)

    # Branch path
    branch1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu')(input_layer)
    branch2 = AvgPool2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    branch3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(input_layer)

    # Concatenate main path and branch path
    main_path_output = Concatenate()([conv4, branch1, branch2, branch3])

    # Apply 1x1 convolution
    conv5 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), activation='relu')(main_path_output)

    # Flatten
    flatten = Flatten()(conv5)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model