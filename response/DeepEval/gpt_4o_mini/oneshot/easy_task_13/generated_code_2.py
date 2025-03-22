import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():     
    input_layer = Input(shape=(28, 28, 1))

    # Two 1x1 convolutional layers with dropout
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    drop1 = Dropout(0.3)(conv1)

    conv2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(drop1)
    drop2 = Dropout(0.3)(conv2)

    # 3x1 convolutional layer with dropout
    conv3 = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(drop2)
    drop3 = Dropout(0.3)(conv3)

    # 1x3 convolutional layer with dropout
    conv4 = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(drop3)
    drop4 = Dropout(0.3)(conv4)

    # 1x1 convolutional layer to restore original channel count
    conv5 = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(drop4)

    # Adding the processed features with the original input
    added_output = Add()([input_layer, conv5])

    # Flattening layer followed by a fully connected layer
    flatten_layer = Flatten()(added_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model