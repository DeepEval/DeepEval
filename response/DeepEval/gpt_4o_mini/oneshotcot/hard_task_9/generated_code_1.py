import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense, BatchNormalization
from keras.models import Model

def dl_model():
    
    # Step 1: Add Input Layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels
    
    # Step 2: Define the first branch with 1x1 convolution
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 3: Define the second branch with 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)

    # Step 4: Define the third branch with 1x1 convolution followed by two 3x3 convolutions
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(branch3)

    # Step 5: Concatenate the outputs of the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Step 6: 1x1 convolution to adjust the output dimensions
    conv_output = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(concatenated)

    # Step 7: Add the main path with the input (skip connection)
    main_path = Add()([conv_output, input_layer])

    # Step 8: Apply batch normalization
    bath_norm = BatchNormalization()(main_path)

    # Step 9: Flatten the result
    flatten_layer = Flatten()(bath_norm)

    # Step 10: Add three fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    # Step 11: Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model