import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First 1x1 convolutional layer with dropout
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1_1 = Dropout(0.3)(conv1_1)

    # Second 1x1 convolutional layer with dropout
    conv1_2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(dropout1_1)
    dropout1_2 = Dropout(0.3)(conv1_2)

    # 3x1 convolutional layer
    conv3_1 = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(dropout1_2)

    # 1x3 convolutional layer
    conv1_3 = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(dropout1_2)

    # 1x1 convolutional layer to match input channels
    conv1x1_restore = Conv2D(filters=1, kernel_size=(1, 1), padding='same')(Add()([conv3_1, conv1_3]))

    # Add the processed features with the original input
    added_output = Add()([input_layer, conv1x1_restore])

    # Flattening layer
    flatten_layer = Flatten()(added_output)

    # Fully connected layer to produce final probability distribution
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model