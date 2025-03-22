from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, concatenate, Flatten, Dense

def dl_model():
    # Define input shape
    input_shape = (32, 32, 3)

    # Define three blocks of convolutional operations
    block1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_shape)
    block2 = Conv2D(128, (3, 3), activation='relu', padding='same')(block1)
    block3 = Conv2D(256, (3, 3), activation='relu', padding='same')(block2)

    # Define batch normalization and ReLU activation functions
    bn1 = BatchNormalization()(block1)
    bn2 = BatchNormalization()(block2)
    bn3 = BatchNormalization()(block3)
    relu1 = ReLU()(bn1)
    relu2 = ReLU()(bn2)
    relu3 = ReLU()(bn3)

    # Define concatenation of outputs from each block
    x = concatenate([relu1, relu2, relu3], axis=1)

    # Define fully connected layers
    flatten = Flatten()(x)
    dense1 = Dense(128, activation='relu')(flatten)
    dense2 = Dense(10, activation='softmax')(dense1)

    # Define model
    model = Model(input_shape, dense2)

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model