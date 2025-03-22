import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # First 1x1 convolutional layer with dropout
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv1 = Dropout(0.2)(conv1)

    # Second 1x1 convolutional layer with dropout
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(conv1)
    conv2 = Dropout(0.2)(conv2)

    # 3x1 convolutional layer with dropout
    conv3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='valid', activation='relu')(conv2)
    conv3 = Dropout(0.2)(conv3)

    # 1x3 convolutional layer with dropout
    conv4 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='valid', activation='relu')(conv3)
    conv4 = Dropout(0.2)(conv4)

    # 1x1 convolutional layer to restore channel count
    conv5 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(conv4)

    # Add processed features to original input
    add_features = Add()([conv5, input_layer])

    # Flatten and fully connected layer
    flatten_layer = Flatten()(add_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model