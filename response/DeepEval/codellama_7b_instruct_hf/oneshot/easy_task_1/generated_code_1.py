from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D((2, 2))(conv2)
    flatten = Flatten()(max_pooling2)
    dense1 = Dense(128, activation='relu')(flatten)
    dense2 = Dense(64, activation='relu')(dense1)
    output = Dense(10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output)

    return model