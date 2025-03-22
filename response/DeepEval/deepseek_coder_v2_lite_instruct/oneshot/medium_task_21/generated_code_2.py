import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def branch1(input_tensor):
        x = Conv2D(64, (1, 1), padding='same', activation='relu')(input_tensor)
        return x

    def branch2(input_tensor):
        x = Conv2D(64, (1, 1), padding='same', activation='relu')(input_tensor)
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        return x

    def branch3(input_tensor):
        x = Conv2D(64, (1, 1), padding='same', activation='relu')(input_tensor)
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        return x

    def branch4(input_tensor):
        x = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input_tensor)
        x = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
        return x

    # Apply branches to input
    branch1_output = branch1(input_layer)
    branch2_output = branch2(input_layer)
    branch3_output = branch3(input_layer)
    branch4_output = branch4(input_layer)

    # Concatenate outputs from all branches
    concat_layer = Concatenate()([branch1_output, branch2_output, branch3_output, branch4_output])

    # Batch Normalization
    batch_norm = BatchNormalization()(concat_layer)

    # Flatten the result
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(256, activation='relu')(flatten_layer)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(128, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)
    output_layer = Dense(10, activation='softmax')(dropout2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model