import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Add

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    dropout1 = Dropout(0.2)(conv1)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(dropout1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling1)
    dropout2 = Dropout(0.2)(conv2)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(dropout2)

    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling2)

    branch_input = Input(shape=(28, 28, 1))
    branch_output = conv3

    # Adding the branch to the main path
    combined = Add()([max_pooling2, branch_output])

    bath_norm = BatchNormalization()(combined)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=[input_layer, branch_input], outputs=output_layer)

    return model