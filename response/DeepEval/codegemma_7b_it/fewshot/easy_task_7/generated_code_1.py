import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    conv1_3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_2)
    dropout1 = Dropout(rate=0.25)(conv1_3)
    conv2_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout1)
    conv2_2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)
    conv2_3 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_2)
    dropout2 = Dropout(rate=0.25)(conv2_3)
    conv3 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout2)

    # Branch path
    conv4 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv5 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv4)

    # Combining outputs
    adding_layer = Add()([conv3, conv5])

    # Final layers
    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Model definition
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model