import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense

def dl_model():     
    
    input_layer = Input(shape=(32, 32, 3))

    x = Lambda(lambda x: tf.split(x, 3, axis=2))(input_layer)

    # Group 1: 1x1 convolution
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])

    # Group 2: 3x3 convolution
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])

    # Group 3: 5x5 convolution
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])

    # Concatenate the outputs
    merged = Concatenate()( [conv1, conv2, conv3] )

    # Flatten and fully connected layer
    flatten_layer = Flatten()(merged)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model