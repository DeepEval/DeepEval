import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    input_img = Input(shape=(32, 32, 3))

    # Branch 1: 3x3 convolutions
    conv_1x1_branch_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    conv_3x3_branch_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv_1x1_branch_1)

    # Branch 2: 1x1 followed by 3x3 convolutions
    conv_1x1_branch_2 = Conv2D(16, (1, 1), activation='relu', padding='same')(input_img)
    conv_3x3_branch_2 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv_1x1_branch_2)

    # Branch 3: Max pooling
    pool_branch_3 = MaxPooling2D((2, 2), strides=(2, 2))(input_img)

    # Feature fusion
    concat = Concatenate()([conv_1x1_branch_1, conv_3x3_branch_1, conv_3x3_branch_2, pool_branch_3])

    # Fully connected layers
    flatten = Flatten()(concat)
    dense_1 = Dense(500, activation='relu')(flatten)
    dense_2 = Dense(10, activation='softmax')(dense_1)

    # Model creation
    model = Model(inputs=input_img, outputs=dense_2)

    return model