import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Concatenate, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Branch 1
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1)
    
    # Branch 2
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch2)
    branch2 = Lambda(lambda x: tf.image.resize(x, (32, 32)))(branch2)
    
    # Branch 3
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = Lambda(lambda x: tf.image.resize(x, (32, 32)))(branch3)
    
    # Concatenate outputs from all branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Final 1x1 convolutional layer
    main_path_output = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(concatenated)

    # Branch path
    branch_path_input = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    branch_path_output = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch_path_input)

    # Add outputs from both paths
    added = Add()([main_path_output, branch_path_output])

    # Flatten and pass through two fully connected layers
    flattened = Flatten()(added)
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model