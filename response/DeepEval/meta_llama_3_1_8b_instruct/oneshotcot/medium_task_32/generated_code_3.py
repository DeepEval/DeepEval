import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Reshape, DepthwiseConv2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the last dimension
    split_input = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Define three groups of feature extraction via depthwise separable convolutional layers
    group1 = DepthwiseConv2D(kernel_size=(1, 1), activation='relu')(split_input[0])
    group2 = DepthwiseConv2D(kernel_size=(3, 3), activation='relu')(split_input[1])
    group3 = DepthwiseConv2D(kernel_size=(5, 5), activation='relu')(split_input[2])
    
    # Concatenate the outputs of the three groups
    output_tensor = Concatenate()([group1, group2, group3])
    
    # Reshape the output to match the input shape of the dense layer
    output_tensor = Reshape((output_tensor.shape[1], output_tensor.shape[2], 1))(output_tensor)
    
    # Add a convolutional layer to increase the number of features
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(output_tensor)
    
    # Flatten the output
    flatten_layer = Flatten()(conv)
    
    # Add a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model