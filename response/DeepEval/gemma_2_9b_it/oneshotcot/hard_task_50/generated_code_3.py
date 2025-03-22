import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Dense, Lambda, Conv2D, Reshape, Concatenate
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # First Block
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    flat1 = Flatten()(pool1)
    
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    flat2 = Flatten()(pool2)

    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    flat3 = Flatten()(pool3)

    dropout_out = Dropout(0.25)(Concatenate()([flat1, flat2, flat3]))
    dense1 = Dense(units=128, activation='relu')(dropout_out)
    reshape_layer = Reshape((1, 128))(dense1) 

    # Second Block
    split_tensor = Lambda(lambda x: tf.split(x, 4, axis=1))(reshape_layer)
    
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
    conv4 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(split_tensor[3])
    
    concatenate_output = Concatenate()( [conv1, conv2, conv3, conv4])

    flatten_final = Flatten()(concatenate_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_final)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model