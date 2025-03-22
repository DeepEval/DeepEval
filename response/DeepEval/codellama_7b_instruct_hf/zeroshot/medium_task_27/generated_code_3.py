import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Softmax
from keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Convolutional branch 1
    branch1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    branch1 = Conv2D(64, (5, 5), activation='relu')(branch1)
    branch1 = MaxPooling2D((2, 2))(branch1)

    # Convolutional branch 2
    branch2 = Conv2D(32, (3, 3), activation='relu')(inputs)
    branch2 = Conv2D(64, (5, 5), activation='relu')(branch2)
    branch2 = MaxPooling2D((2, 2))(branch2)

    # Combine branches
    branch_concat = keras.layers.concatenate([branch1, branch2], axis=1)

    # Global average pooling
    x = GlobalAveragePooling2D()(branch_concat)

    # Dense layers
    x = Dense(256, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Final output
    outputs = Dense(10, activation='softmax')(x)

    # Define model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model