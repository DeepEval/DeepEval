import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_tensor = Input(shape=(32, 32, 3))

    # Branch 1
    branch1_1 = Conv2D(32, (1, 1), activation='relu')(input_tensor)
    branch1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch1_1)
    branch1_dropout = Dropout(0.2)(branch1_2)

    # Branch 2
    branch2_1 = Conv2D(32, (1, 1), activation='relu')(input_tensor)
    branch2_2 = Conv2D(64, (1, 7), activation='relu', padding='same')(branch2_1)
    branch2_3 = Conv2D(64, (7, 1), activation='relu', padding='same')(branch2_2)
    branch2_4 = Conv2D(128, (3, 3), activation='relu', padding='same')(branch2_3)
    branch2_dropout = Dropout(0.2)(branch2_4)

    # Branch 3
    branch3 = MaxPooling2D((2, 2), padding='same')(input_tensor)
    branch3_conv = Conv2D(64, (3, 3), activation='relu', padding='same')(branch3)
    branch3_dropout = Dropout(0.2)(branch3_conv)

    # Concatenate branches
    concatenated = tf.keras.layers.concatenate([branch1_dropout, branch2_dropout, branch3_dropout], axis=-1)

    # Flatten and fully connected layers
    flatten = Flatten()(concatenated)
    dense1 = Dense(512, activation='relu')(flatten)
    dense2 = Dense(256, activation='relu')(dense1)
    output = Dense(10, activation='softmax')(dense2)

    # Create model
    model = Model(inputs=input_tensor, outputs=output)
    return model