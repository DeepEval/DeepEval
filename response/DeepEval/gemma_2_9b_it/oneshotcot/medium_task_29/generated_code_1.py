import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Convolutional block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(conv1)
    flat1 = Flatten()(pool1)

    # Convolutional block 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(flat1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    flat2 = Flatten()(pool2)

    # Convolutional block 3
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(flat2)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(conv3)
    flat3 = Flatten()(pool3)

    # Concatenate feature maps
    concat_features = Concatenate()([flat1, flat2, flat3])

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concat_features)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model