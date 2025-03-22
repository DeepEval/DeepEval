import keras
from keras.layers import Input, Conv2D, Add, Lambda, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split the input layer into three groups along the channel dimension
    def split_input(input_tensor):
        return tf.split(input_tensor, num_or_size_splits=3, axis=-1)
    
    split_layer = Lambda(split_input)(input_layer)

    # Define a series of convolutions for each group
    def conv_group(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        return conv3
    
    # Apply the series of convolutions to each group
    group1 = conv_group(split_layer[0])
    group2 = conv_group(split_layer[1])
    group3 = conv_group(split_layer[2])

    # Combine the outputs from the three groups using an addition operation
    main_path = Add()([group1, group2, group3])

    # Fuse the main path with the original input layer
    combined_path = Add()([main_path, input_layer])

    # Flatten the combined path into a one-dimensional vector
    flatten_layer = Flatten()(combined_path)

    # Define a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model