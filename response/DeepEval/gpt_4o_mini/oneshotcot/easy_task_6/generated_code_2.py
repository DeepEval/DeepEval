import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():     
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))

    # Step 2: Main path
    # First convolutional layer that increases the feature map width
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    # Second convolutional layer that restores the number of channels
    conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Step 3: Branch path directly connects to input
    branch_path = input_layer

    # Step 4: Combine both paths with addition
    combined = Add()([conv2, branch_path])

    # Step 5: Add flatten layer
    flatten_layer = Flatten()(combined)

    # Step 6: Add a fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model