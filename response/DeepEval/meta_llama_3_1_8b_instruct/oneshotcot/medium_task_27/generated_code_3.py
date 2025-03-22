import keras
from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense, Multiply

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first convolutional branch with a 3x3 kernel
    conv_branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Define the second convolutional branch with a 5x5 kernel
    conv_branch2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine the output of the two branches through addition
    combined_output = Add()([conv_branch1, conv_branch2])

    # Define a global average pooling layer to compress the features
    global_pool = GlobalAveragePooling2D()(combined_output)

    # Define the first fully connected layer with a softmax activation function
    dense1 = Dense(units=128, activation='relu')(global_pool)
    
    # Define the second fully connected layer with a softmax activation function
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Define a weighted sum of the output of each branch
    weight1 = Multiply()([conv_branch1, dense1])
    weight2 = Multiply()([conv_branch2, dense1])
    weighted_output = Add()([weight1, weight2])

    # Define the final fully connected layer with a softmax activation function
    output_layer = Dense(units=10, activation='softmax')(weighted_output)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model