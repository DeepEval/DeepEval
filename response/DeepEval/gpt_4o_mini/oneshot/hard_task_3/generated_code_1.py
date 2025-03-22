import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, Lambda, Concatenate, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels and have 3 color channels

    # Split the input into 3 groups along the channel dimension
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Main pathway: Process each split through 1x1 and 3x3 convolutions
    def main_pathway(split_input):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_input)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        return Dropout(0.25)(conv2)  # Apply dropout

    # Process each of the split inputs
    group1 = main_pathway(split_inputs[0])
    group2 = main_pathway(split_inputs[1])
    group3 = main_pathway(split_inputs[2])

    # Concatenate the outputs from the three groups to form the main pathway
    main_output = Concatenate()([group1, group2, group3])

    # Branch pathway: Process the original input through a 1x1 convolution
    branch_output = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the main and branch pathways using an addition operation
    combined_output = Add()([main_output, branch_output])

    # Flatten the combined output and apply a fully connected layer for classification
    flatten_layer = Flatten()(combined_output)
    dense_output = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=dense_output)

    return model