import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    branch1 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch1 = Conv2D(64, (3, 3), activation='relu')(branch1)
    branch1 = Dropout(0.2)(branch1)

    branch2 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(64, (1, 7), activation='relu')(branch2)
    branch2 = Conv2D(64, (7, 1), activation='relu')(branch2)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)
    branch2 = Dropout(0.2)(branch2)

    branch3 = MaxPooling2D(pool_size=(2, 2))(input_layer)
    branch3 = Dropout(0.2)(branch3)

    merged_outputs = keras.layers.concatenate([branch1, branch2, branch3])

    flattened_outputs = Flatten()(merged_outputs)

    output_layer = Dense(10, activation='softmax')(flattened_outputs)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model