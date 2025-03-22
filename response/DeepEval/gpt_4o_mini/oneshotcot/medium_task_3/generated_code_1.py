import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: First convolutional layer followed by max pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    # Step 3: Second convolutional layer followed by max pooling
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Ensure the output dimensions match the input channel dimension
    conv2_output = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')(max_pool2)

    # Step 4: Combine the outputs with the input through an addition operation
    combined = Add()([input_layer, conv2_output])

    # Step 5: Flatten the result and pass through a dense layer for classification
    flatten_layer = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Step 6: Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model