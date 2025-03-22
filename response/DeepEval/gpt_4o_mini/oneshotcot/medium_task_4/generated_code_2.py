import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: Two blocks of convolution followed by average pooling
    path1_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path1_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1_conv1)
    path1_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(path1_conv2)
    
    # Path 2: Single convolutional layer
    path2_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine both paths using addition
    combined_output = Add()([path1_pool, path2_conv])

    # Flatten the combined output
    flatten_layer = Flatten()(combined_output)
    
    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model