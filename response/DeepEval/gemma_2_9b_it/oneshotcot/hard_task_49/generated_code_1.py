import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Lambda, tf

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    # First Block: Average Pooling
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)
    
    concat_pool = Concatenate()([flat1, flat2, flat3])

    dense1 = Dense(units=128, activation='relu')(concat_pool)
    reshape_layer = keras.layers.Reshape((1, 128))(dense1)
    

    # Second Block: Depthwise Separable Convolutions
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=1))(reshape_layer)
    
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layer[2])
    conv4 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(split_layer[3])
    
    concat_conv = Concatenate()([conv1, conv2, conv3, conv4])
    
    flat_final = Flatten()(concat_conv)
    output_layer = Dense(units=10, activation='softmax')(flat_final)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model