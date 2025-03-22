from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
       
    path1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2,2), padding='same', activation='relu')(input_layer)
    
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1,1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(conv1)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2,2), padding='same', activation='relu')(conv2)
    
    path3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_layer)

    concated = Concatenate()([path1, path2, path3])
    flatten_output = Flatten()(concated)
    dense_1 = Dense(units=128, activation='relu')(flatten_output)
    output_layer = Dense(units=10, activation='softmax')(dense_1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model