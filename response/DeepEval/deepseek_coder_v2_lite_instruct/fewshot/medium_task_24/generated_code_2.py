import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv1_3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1_1)
    branch1 = Dropout(0.5)(conv1_3)

    # Branch 2
    conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv2_7_1 = Conv2D(filters=64, kernel_size=(1, 7), activation='relu')(conv2_1)
    conv2_1_7 = Conv2D(filters=64, kernel_size=(7, 1), activation='relu')(conv2_7_1)
    conv2_3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv2_1_7)
    branch2 = Dropout(0.5)(conv2_3)

    # Branch 3
    pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    branch3 = Dropout(0.5)(pool)

    # Concatenate outputs from all branches
    concat = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated output
    flatten = Flatten()(concat)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model