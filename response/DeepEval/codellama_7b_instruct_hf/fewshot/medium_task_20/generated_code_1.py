import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first convolutional path
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    conv1_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_2)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_3)

    # Define the second convolutional path
    conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2_2)

    # Define the third convolutional path
    conv3_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_1)
    max_pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3_2)

    # Define the fourth convolutional path
    conv4_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv4_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4_1)
    max_pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4_2)

    # Merge the outputs of the convolutional paths
    merge = Add()([max_pool1, max_pool2, max_pool3, max_pool4])

    # Flatten the merged output and add a dense layer with 128 units
    flatten = Flatten()(merge)
    dense = Dense(units=128, activation='relu')(flatten)

    # Add a final dense layer with 10 units and softmax activation for classification
    output_layer = Dense(units=10, activation='softmax')(dense)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model