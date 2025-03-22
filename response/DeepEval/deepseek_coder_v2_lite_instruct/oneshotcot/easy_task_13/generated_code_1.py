import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First 1x1 convolutional layer
    conv1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    dropout1 = Dropout(0.25)(conv1x1_1)

    # Second 1x1 convolutional layer
    conv1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    dropout2 = Dropout(0.25)(conv1x1_2)

    # 3x1 convolutional layer
    conv3x1 = Conv2D(filters=32, kernel_size=(3, 1), activation='relu')(conv1x1_1)
    dropout3 = Dropout(0.25)(conv3x1)

    # 1x3 convolutional layer
    conv1x3 = Conv2D(filters=32, kernel_size=(1, 3), activation='relu')(conv1x1_2)
    dropout4 = Dropout(0.25)(conv1x3)

    # Restore channels to match input
    restore_channels = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(dropout3)
    restore_channels = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(dropout4)

    # Add processed features with the original input
    added = Add()([restore_channels, input_layer])

    # Flatten the result
    flatten_layer = Flatten()(added)

    # Fully connected layer
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model