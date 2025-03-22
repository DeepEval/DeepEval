from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense
from keras.layers import Layer

def dl_model():
    # Input layer
    input_A = Input(shape=(28, 28, 1))
    input_B = Input(shape=(28, 28, 1))

    # First processing pathway
    x = input_A
    for _ in range(3):
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2))(x)

    # Second processing pathway
    y = input_B
    for _ in range(3):
        y = Conv2D(64, (3, 3), activation='relu')(y)
        y = BatchNormalization()(y)
        y = ReLU()(y)
        y = Conv2D(64, (3, 3), activation='relu')(y)
        y = BatchNormalization()(y)
        y = ReLU()(y)
        y = MaxPooling2D((2, 2))(y)

    # Concatenate features from both pathways
    concat = Concatenate(axis=-1)([x, y])

    # Flatten and classify
    x = Flatten()(concat)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)

    # Classification head
    output = Dense(10, activation='softmax')(x)  # Assuming 10 classes for MNIST

    # Model
    model = Model(inputs=[input_A, input_B], outputs=output)
    return model

# To use the model, you can call:
# model = dl_model()
# model.summary()