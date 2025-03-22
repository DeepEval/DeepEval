import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Concatenate, Dense, Reshape

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)
    drop1 = Dropout(0.25)(flat1)
    drop2 = Dropout(0.25)(flat2)
    drop3 = Dropout(0.25)(flat3)
    concat1 = Concatenate()([drop1, drop2, drop3])

    # Fully connected layer and reshaping
    fc1 = Dense(128, activation='relu')(concat1)
    reshape_layer = Reshape((1, 128))(fc1) 

    # Block 2
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv2_2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv3_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv3_2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv3_3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    pool4_1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(reshape_layer)
    conv4_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool4_1)

    concat2 = Concatenate()([conv1_1, conv2_1, conv2_2, conv3_1, conv3_2, conv3_3, conv4_1])

    # Final classification layers
    flatten_layer = Flatten()(concat2)
    dense4 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense4)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model