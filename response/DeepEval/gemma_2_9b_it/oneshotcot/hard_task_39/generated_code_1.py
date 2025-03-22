import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Concatenate, Dense, Reshape

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    flat1 = Flatten()(pool1)
    
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    flat2 = Flatten()(pool2)
    
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    flat3 = Flatten()(pool3)

    concat_block1 = Concatenate()([flat1, flat2, flat3])
    dense1 = Dense(units=128, activation='relu')(concat_block1)
    reshape_layer = Reshape((128, 1, 1))(dense1)  

    # Block 2
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshape_layer)
    
    output_block2 = Concatenate()([path1, path2, path3, path4])
    
    flatten_layer = Flatten()(output_block2)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model