import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main pathway
    main_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch pathway
    branch_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_conv2 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_conv3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Concatenate the outputs from the branch path
    branch_output = Concatenate()([branch_conv1, branch_conv2, branch_conv3])

    # Combine both paths
    combined_output = Concatenate()([main_conv1, branch_output])
    
    # Final 1x1 convolution to reduce dimensionality
    final_conv = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(combined_output)

    # Add the direct connection from input
    added_output = Add()([final_conv, input_layer])

    # Flatten and fully connected layers for classification
    flatten_layer = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model