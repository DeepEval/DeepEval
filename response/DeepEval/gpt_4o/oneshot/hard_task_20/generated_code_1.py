import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def split_and_conv(input_tensor):
        # Split the input into 3 groups
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Process each group with different convolutional layers
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(splits[2])

        # Concatenate the outputs
        concatenated = Concatenate()([conv1, conv2, conv3])
        
        return concatenated
    
    main_path_output = split_and_conv(input_layer)

    # Branch path
    branch_conv = Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the outputs of main and branch paths
    fused_features = Add()([main_path_output, branch_conv])

    # Classification through fully connected layers
    flatten_layer = Flatten()(fused_features)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model