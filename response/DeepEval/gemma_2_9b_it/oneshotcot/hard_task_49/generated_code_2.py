import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Lambda, tf
from keras.layers.experimental import SeparableConv2D

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # First Block: Average Pooling Layers
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)
    concatenated_pool = Concatenate()([flat1, flat2, flat3])

    dense1 = Dense(units=128, activation='relu')(concatenated_pool)
    reshape_layer = keras.layers.Reshape((1, 128))(dense1)

    # Second Block: Depthwise Separable Convolutions
    split_layer = Lambda(lambda x: tf.split(x, 4, axis=3))(reshape_layer)
    
    conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_layer[0])
    conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu')(split_layer[1])
    conv3 = SeparableConv2D(filters=128, kernel_size=(5, 5), activation='relu')(split_layer[2])
    conv4 = SeparableConv2D(filters=256, kernel_size=(7, 7), activation='relu')(split_layer[3])
    
    concatenated_conv = Concatenate(axis=3)([conv1, conv2, conv3, conv4])
    flatten_output = Flatten()(concatenated_conv)
    output_layer = Dense(units=10, activation='softmax')(flatten_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model