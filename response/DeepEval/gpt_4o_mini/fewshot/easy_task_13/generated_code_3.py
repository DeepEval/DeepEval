import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Two 1x1 convolutional layers with dropout
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(0.25)(conv1)
    
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(0.25)(conv2)

    # A 3x1 convolutional layer
    conv3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(dropout2)

    # A 1x3 convolutional layer
    conv4 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(dropout2)

    # Restore the channel count with another 1x1 convolutional layer
    conv5 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3)
    
    # Combine the processed features with the original input via addition
    added_layer = Add()([conv5, input_layer])

    # Flatten the result and apply a fully connected layer
    flatten_layer = Flatten()(added_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model