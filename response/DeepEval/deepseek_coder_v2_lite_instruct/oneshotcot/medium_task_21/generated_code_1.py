import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    def branch1(input_tensor):
        x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return x

    def branch2(input_tensor):
        x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        return x

    def branch3(input_tensor):
        x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        return x

    def branch4(input_tensor):
        x = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        return x

    # Apply branches to input layer
    branch1_output = branch1(input_layer)
    branch2_output = branch2(input_layer)
    branch3_output = branch3(input_layer)
    branch4_output = branch4(input_layer)

    # Concatenate outputs of all branches
    concat_output = Concatenate()([branch1_output, branch2_output, branch3_output, branch4_output])

    # Batch normalization
    batch_norm = BatchNormalization()(concat_output)

    # Flatten the result
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(units=128, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)
    output_layer = Dense(units=10, activation='softmax')(dropout2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model