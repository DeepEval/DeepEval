import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    
    # Define the input layer with shape (32, 32, 3)
    input_layer = Input(shape=(32, 32, 3))

    # Step 1: Define the first convolutional layer
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 2: Define the second convolutional layer, concatenate the output with the input of the previous layer
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(Concatenate()([conv1, input_layer]))

    # Step 3: Define the third convolutional layer, concatenate the output with the input of the previous layer
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(Concatenate()([conv2, conv1]))

    # Step 4: Flatten the output of the convolutional layers
    flatten_layer = Flatten()(conv3)

    # Step 5: Define the first fully connected layer
    dense1 = Dense(units=512, activation='relu')(flatten_layer)

    # Step 6: Define the second fully connected layer
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model