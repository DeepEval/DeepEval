import keras
from keras.layers import Input, Conv2D, Dropout, Concatenate, MaxPooling2D, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Main Path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    dropout1 = Dropout(0.2)(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(conv2)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(dropout2)

    # Branch Path
    branch_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Combine paths
    concat = Concatenate()([conv3, branch_conv])

    # Post-combination layers
    add = Add()([concat, branch_conv])
    flatten = Flatten()(add)
    dense = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model