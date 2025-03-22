import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Dense, Reshape

def dl_model():  
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    flat1 = Flatten()(pool1)
    dropout1 = Dropout(0.2)(flat1)
    flat2 = Flatten()(pool2)
    dropout2 = Dropout(0.2)(flat2)
    flat3 = Flatten()(pool3)
    dropout3 = Dropout(0.2)(flat3)

    block1_output = Concatenate()([dropout1, dropout2, dropout3])

    # Fully connected and reshape for Block 2
    dense1 = Dense(128, activation='relu')(block1_output)
    reshape_layer = Reshape((1, 128))(dense1)

    # Block 2
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv2_2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv3_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv3_2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv3_3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    pool4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(reshape_layer)
    conv4_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool4)

    concat_output = Concatenate(axis=1)([conv1_1, conv2_1, conv2_2, conv3_1, conv3_2, conv3_3, conv4_1])

    # Final layers
    flatten_layer = Flatten()(concat_output)
    dense2 = Dense(64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model