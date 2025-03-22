import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1
    conv1_1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv1_3_3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1_1_1)
    branch1_output = Dropout(0.25)(conv1_3_3)

    # Branch 2
    conv2_1_1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    conv2_1_7 = Conv2D(filters=64, kernel_size=(1, 7), activation='relu')(input_layer)
    conv2_7_1 = Conv2D(filters=64, kernel_size=(7, 1), activation='relu')(input_layer)
    conv2_3_3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(conv2_1_7)
    branch2_output = Dropout(0.25)(conv2_3_3)

    # Branch 3
    pool_output = MaxPooling2D(pool_size=(2, 2))(input_layer)
    branch3_output = Dropout(0.25)(pool_output)

    # Concatenate branches
    merged_output = Concatenate()([branch1_output, branch2_output, branch3_output])

    # Fully connected layers
    flatten_layer = Flatten()(merged_output)
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model