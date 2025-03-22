import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # First branch
    branch_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    branch_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(branch_1)
    branch_1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_1)
    branch_1 = BatchNormalization()(branch_1)

    # Second branch
    branch_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    branch_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(branch_2)
    branch_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_2)
    branch_2 = BatchNormalization()(branch_2)

    # Third branch
    branch_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    branch_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(branch_3)
    branch_3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_3)
    branch_3 = BatchNormalization()(branch_3)

    # Concatenate branches
    branches = Concatenate()([branch_1, branch_2, branch_3])

    # Flatten and add fully connected layers
    branches = Flatten()(branches)
    branches = Dense(units=128, activation='relu')(branches)
    branches = Dense(units=64, activation='relu')(branches)
    branches = Dense(units=10, activation='softmax')(branches)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=branches)

    return model