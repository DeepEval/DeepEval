import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Dropout, Flatten, Dense, BatchNormalization
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 1x1 Convolution
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Dropout(0.2)(branch1)

    # Branch 2: 1x1 Convolution followed by 3x3 Convolution
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
    branch2 = Dropout(0.2)(branch2)

    # Branch 3: 1x1 Convolution followed by two consecutive 3x3 Convolutions
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
    branch3 = Dropout(0.2)(branch3)

    # Branch 4: Average Pooling followed by 1x1 Convolution
    branch4 = AveragePooling2D(pool_size=(2, 2))(input_layer)
    branch4 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(branch4)
    branch4 = Dropout(0.2)(branch4)

    # Concatenate all branches
    concatenated = Concatenate()([branch1, branch2, branch3, branch4])

    # Batch normalization
    batch_norm = BatchNormalization()(concatenated)

    # Flatten and Dense layers
    flatten_layer = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model