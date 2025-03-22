import keras
from keras.layers import Input, Conv2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Step 1: 1x1 Convolution to reduce dimensionality
    reduced_dim = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 2: 3x3 Convolution to extract features
    features = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reduced_dim)

    # Step 3: 1x1 Convolution to restore dimensionality
    restored_dim = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(features)

    # Step 4: Flatten the output
    flatten_output = Flatten()(restored_dim)

    # Step 5: Fully connected layer with 10 neurons for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_output)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model