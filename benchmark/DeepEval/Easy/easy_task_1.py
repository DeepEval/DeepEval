from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))
    conv_1 = Conv2D(filters=6, kernel_size=(5, 5), activation='tanh', padding='valid')(input_layer)
    pool_1 = MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(conv_1)
    conv_2 = Conv2D(filters=16, kernel_size=(5, 5), activation='tanh', padding='valid')(pool_1)
    pool_2 = MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(conv_2)

    flatten = Flatten()(pool_2)
    dense_1 = Dense(units=120, activation='tanh')(flatten)
    dense_2 = Dense(units=84, activation='tanh')(dense_1)
    output_layer = Dense(units=10, activation='softmax')(dense_2)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model


