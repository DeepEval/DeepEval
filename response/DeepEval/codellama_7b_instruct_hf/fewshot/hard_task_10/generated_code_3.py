import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense
 和 return model
def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first feature extraction path: 1x1 convolution
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_shape)

    # Define the second feature extraction path: sequence of convolutions: 1x1, followed by 1x7, and then 7x1
    conv1_2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    conv1_3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv1_2)

    # Define the main path: concatenate the outputs of the two feature extraction paths
    main_path = Add()([conv1_1, conv1_3])

    # Define the branch path: merge the output of the main path with the input
    branch_path = Add()([main_path, input_shape])

    # Define the final classification layer
    output_layer = Dense(units=10, activation='softmax')(branch_path)

    # Create the model
    model = keras.Model(inputs=input_shape, outputs=output_layer)

    return model