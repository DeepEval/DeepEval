import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense

def dl_model():     
    # Define the input layer with shape (28, 28, 1) for the MNIST dataset
    input_layer = Input(shape=(28, 28, 1))

    # Define the first block of the model
    def local_feature_block(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
        dropout_layer = Dropout(0.2)(avg_pool)
        return dropout_layer
    
    # Apply two consecutive blocks
    block_output = local_feature_block(input_layer)
    block_output = local_feature_block(block_output)

    # Apply global average pooling to convert high-dimensional feature maps into a one-dimensional vector
    global_avg_pool = GlobalAveragePooling2D()(block_output)

    # Flatten the output of the global average pooling layer
    flatten_layer = Flatten()(global_avg_pool)

    # Define a fully connected layer to produce the final classification output
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model with the specified input and output layers
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model