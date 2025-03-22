from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu')(maxpool2)
    maxpool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    flatten = Flatten()(maxpool3)

    dense1 = Dense(128, activation='relu')(flatten)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model