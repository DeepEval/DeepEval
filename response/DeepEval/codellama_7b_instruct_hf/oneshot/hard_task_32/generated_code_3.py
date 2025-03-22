import keras
from keras.layers import Input, Conv2D, MaxPooling2D, DepthwiseSeparableConv2D, BatchNormalization, Flatten, Dense
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First branch
    branch_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    branch_1 = Dropout(0.2)(branch_1)
    branch_1 = Conv2D(64, (1, 1), activation='relu', padding='same')(branch_1)
    branch_1 = Dropout(0.2)(branch_1)
    branch_1 = BatchNormalization()(branch_1)
    branch_1 = Flatten()(branch_1)

    # Second branch
    branch_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    branch_2 = Dropout(0.2)(branch_2)
    branch_2 = Conv2D(64, (1, 1), activation='relu', padding='same')(branch_2)
    branch_2 = Dropout(0.2)(branch_2)
    branch_2 = BatchNormalization()(branch_2)
    branch_2 = Flatten()(branch_2)

    # Third branch
    branch_3 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    branch_3 = Dropout(0.2)(branch_3)
    branch_3 = Conv2D(64, (1, 1), activation='relu', padding='same')(branch_3)
    branch_3 = Dropout(0.2)(branch_3)
    branch_3 = BatchNormalization()(branch_3)
    branch_3 = Flatten()(branch_3)

    # Concatenate the outputs from all three branches
    merged = Concatenate()([branch_1, branch_2, branch_3])

    # Add fully connected layers
    merged = Dense(128, activation='relu')(merged)
    merged = Dense(64, activation='relu')(merged)
    merged = Dense(10, activation='softmax')(merged)

    model = Model(inputs=input_layer, outputs=merged)

    return model