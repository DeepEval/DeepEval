import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Lambda, tf
from keras.layers.experimental import SeparableConv2D

def dl_model():  
    
    input_layer = Input(shape=(32, 32, 3))  

    # First Block: Max Pooling with Different Scales
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    pool4 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat4 = Flatten()(pool4)

    dropout_layer = keras.layers.Dropout(0.5)(tf.concat([flat1, flat2, flat4], axis=-1))
    
    reshape_layer = keras.layers.Reshape((1, -1))(dropout_layer)
    dense_layer = Dense(units=128, activation='relu')(reshape_layer)

    # Second Block: Separable Convolutions and Feature Extraction
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(dense_layer)
    conv1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(split_layer[0])
    conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(split_layer[1])
    conv3 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')(split_layer[2])
    conv4 = SeparableConv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same')(split_layer[3])

    concatenate_layer = Concatenate()(tf.concat([conv1, conv2, conv3, conv4], axis=-1))

    flatten_output = Flatten()(concatenate_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model