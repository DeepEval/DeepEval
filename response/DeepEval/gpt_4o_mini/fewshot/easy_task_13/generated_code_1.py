import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Dropout
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First 1x1 convolutional layer with dropout
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1_1 = Dropout(0.5)(conv1_1)

    # Second 1x1 convolutional layer with dropout
    conv1_2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(dropout1_1)
    dropout1_2 = Dropout(0.5)(conv1_2)

    # 3x1 convolutional layer with dropout
    conv3_1 = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(dropout1_2)
    dropout3_1 = Dropout(0.5)(conv3_1)

    # 1x3 convolutional layer with dropout
    conv1_3 = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(dropout3_1)
    dropout1_3 = Dropout(0.5)(conv1_3)

    # Restore channels to match input
    conv1x1_out = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(dropout1_3)

    # Combine features with the original input via addition
    added_output = Add()([input_layer, conv1x1_out])

    # Flattening layer
    flatten_layer = Flatten()(added_output)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Constructing the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model