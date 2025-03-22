import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Sequential block 1
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    output1 = Dense(10, activation='softmax')(x)

    # Sequential block 2
    x = Conv2D(64, (3, 3), activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    output2 = Dense(10, activation='softmax')(x)

    # Sequential block 3
    x = Conv2D(128, (3, 3), activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    output3 = Dense(10, activation='softmax')(x)

    # Parallel branch
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    parallel_output = Dense(10, activation='softmax')(x)

    # Add outputs
    x = Add()([output1, output2, output3, parallel_output])

    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Define model
    model = keras.Model(inputs=input_layer, outputs=x)

    return model