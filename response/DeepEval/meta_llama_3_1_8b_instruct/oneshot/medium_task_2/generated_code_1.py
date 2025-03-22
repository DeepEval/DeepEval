import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.layers import Add  # Import the Add layer to combine the outputs of the two paths

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))  # Define the input shape for CIFAR-10 dataset
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    branch_conv = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the two paths using Add layer
    combined_path = Add()([max_pooling, branch_conv])

    bath_norm = BatchNormalization()(combined_path)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    output_layer = dense2  # The final output layer

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model