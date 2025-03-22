import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First block: <convolution, convolution, max pooling>
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

    # Second block: <convolution, convolution, convolution, max pooling>
    conv2_1 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(pool1)
    conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(conv2_1)
    conv2_3 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(conv2_2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_3)

    # Flatten the feature maps
    flatten_layer = Flatten()(pool2)

    # Pass through three fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model