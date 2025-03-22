from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
        
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(input_layer)
    maxpool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_layer)
    concatenated = Concatenate()([conv1, conv2, conv3, maxpool])
    
    flatten_output = Flatten()(concatenated)
    dense_1_output = Dense(units=128, activation='relu')(flatten_output)
    output_layer = Dense(units=10, activation='softmax')(dense_1_output)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
