import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout, Lambda, tf
from keras.applications import ResNet50

def dl_model():
    input_layer = Input(shape=(32, 32, 3)) 

    # First Block
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    pool4 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat4 = Flatten()(pool4)

    dropout = Dropout(0.25)(Concatenate()([flat1, flat2, flat4]))
    dense1 = Dense(128, activation='relu')(dropout)
    reshape_layer = keras.layers.Reshape((1, 128))(dense1) 

    # Second Block
    split_layer = Lambda(lambda x: tf.split(x, 4, axis=-1))(reshape_layer)

    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_layer[0])
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(split_layer[1])
    conv3 = Conv2D(filters=128, kernel_size=(5, 5), activation='relu')(split_layer[2])
    conv4 = Conv2D(filters=256, kernel_size=(7, 7), activation='relu')(split_layer[3])

    concat_layer = Concatenate()( [conv1, conv2, conv3, conv4] )
    flatten_output = Flatten()(concat_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model