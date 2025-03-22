import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    
    # Define the input layer with the CIFAR-10 dataset's image dimensions (32x32x3)
    input_layer = Input(shape=(32, 32, 3))

    # Main path: two blocks of convolution followed by max pooling
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_1)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool1)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_2)

    # Branch path: one block of convolution and max pooling
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2_1)

    # Combine the outputs from both paths using an addition operation
    adding_layer = Add()([main_path, branch_path])

    # Flatten the combined output into a one-dimensional vector
    flatten_layer = Flatten()(adding_layer)

    # Project the features onto a probability distribution across 10 classes
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the Keras model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model