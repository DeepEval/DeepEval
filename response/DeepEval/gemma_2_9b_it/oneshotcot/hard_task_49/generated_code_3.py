import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Concatenate, Dense, Lambda, Reshape, tf

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    # First Block
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)
    
    concat_first_block = Concatenate()([flat1, flat2, flat3])
    
    dense1 = Dense(units=128, activation='relu')(concat_first_block)
    reshape_layer = Reshape((1, 128))(dense1)
    
    # Second Block
    split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(reshape_layer)
    
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
    path3 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
    path4 = Conv2D(filters=256, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(split_tensor[3])
    
    concat_second_block = Concatenate(axis=-1)([path1, path2, path3, path4])
    
    flatten_final = Flatten()(concat_second_block)
    output_layer = Dense(units=10, activation='softmax')(flatten_final)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model