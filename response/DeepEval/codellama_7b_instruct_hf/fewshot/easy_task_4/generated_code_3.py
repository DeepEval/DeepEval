import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First block
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    maxpool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu')(maxpool2)
    maxpool3 = MaxPooling2D((2, 2))(conv3)
    flat1 = Flatten()(maxpool3)

    # Second block
    conv4 = Conv2D(64, (3, 3), activation='relu')(flat1)
    maxpool4 = MaxPooling2D((2, 2))(conv4)
    conv5 = Conv2D(128, (3, 3), activation='relu')(maxpool4)
    maxpool5 = MaxPooling2D((2, 2))(conv5)
    conv6 = Conv2D(256, (3, 3), activation='relu')(maxpool5)
    maxpool6 = MaxPooling2D((2, 2))(conv6)
    flat2 = Flatten()(maxpool6)

    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flat2)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model