import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Create the first 1x1 convolutional layer
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1_1 = Dropout(0.2)(conv1_1)

    # Create the second 1x1 convolutional layer
    conv1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1_1)
    dropout1_2 = Dropout(0.2)(conv1_2)

    # Create the 3x1 convolutional layer
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(dropout1_2)
    dropout2_1 = Dropout(0.2)(conv2_1)

    # Create the 1x3 convolutional layer
    conv2_2 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(dropout2_1)
    dropout2_2 = Dropout(0.2)(conv2_2)

    # Create a 1x1 convolutional layer to match the number of channels
    conv3 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout2_2)

    # Add the original input to the processed features
    adding_layer = Add()([input_layer, conv3])

    # Combine the original input and processed features
    combined_layer = Add()([input_layer, conv3])

    # Flatten the combined layer
    flatten_layer = Flatten()(combined_layer)

    # Create the fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model