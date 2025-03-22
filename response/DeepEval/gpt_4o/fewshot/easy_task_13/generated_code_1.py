import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First 1x1 convolution
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(rate=0.2)(conv1)

    # Second 1x1 convolution
    conv2 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.2)(conv2)

    # 3x1 convolution
    conv3 = Conv2D(filters=16, kernel_size=(3, 1), padding='same', activation='relu')(dropout2)
    dropout3 = Dropout(rate=0.2)(conv3)

    # 1x3 convolution
    conv4 = Conv2D(filters=16, kernel_size=(1, 3), padding='same', activation='relu')(dropout3)
    dropout4 = Dropout(rate=0.2)(conv4)

    # Restore channel count with a final 1x1 convolution
    conv5 = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(dropout4)
    dropout5 = Dropout(rate=0.2)(conv5)

    # Add processed features to the original input
    added_layer = Add()([input_layer, dropout5])

    # Flatten and output fully connected layer
    flatten_layer = Flatten()(added_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model