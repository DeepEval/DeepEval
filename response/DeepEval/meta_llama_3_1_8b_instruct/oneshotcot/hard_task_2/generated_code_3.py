import keras
from keras.layers import Input, Conv2D, Add, Lambda, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 dataset has 3 color channels
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Define the group convolutional block
    def group_conv_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        output_tensor = conv3
        return output_tensor
    
    # Apply the group convolutional block to each group
    group1 = group_conv_block(split[0])
    group2 = group_conv_block(split[1])
    group3 = group_conv_block(split[2])

    # Combine the outputs from the three groups
    add1 = Add()([group1, group2, group3])
    add2 = Add()([add1, input_layer])

    # Flatten the combined features
    flatten_layer = Flatten()(add2)

    # Define the fully connected layer for classification
    dense = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=dense)

    return model