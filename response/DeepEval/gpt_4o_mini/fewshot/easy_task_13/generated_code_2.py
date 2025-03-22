import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First block: two 1x1 convolutional layers with dropout
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(0.3)(conv1_1)
    
    conv1_2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(0.3)(conv1_2)

    # Second block: a 3x1 convolutional layer with dropout
    conv2_1 = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(dropout2)
    dropout3 = Dropout(0.3)(conv2_1)

    # Third block: a 1x3 convolutional layer with dropout
    conv2_2 = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(dropout3)
    dropout4 = Dropout(0.3)(conv2_2)

    # Restore channels with a 1x1 convolutional layer
    conv3 = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(dropout4)

    # Combine processed features with the original input via addition
    combined = Add()([input_layer, conv3])

    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model