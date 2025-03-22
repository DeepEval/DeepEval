import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define input shape
    input_shape = (28, 28, 1)

    # Define first processing pathway
    x1 = Input(shape=input_shape)
    x1 = Conv2D(32, (3, 3), activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(32, (3, 3), activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(32, (3, 3), activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Flatten()(x1)

    # Define second processing pathway
    x2 = Input(shape=input_shape)
    x2 = Conv2D(32, (3, 3), activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(32, (3, 3), activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(32, (3, 3), activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Flatten()(x2)

    # Merge outputs from both pathways
    merged = keras.layers.concatenate([x1, x2], axis=1)

    # Add fully connected layers
    merged = Dense(128, activation='relu')(merged)
    merged = Dense(10, activation='softmax')(merged)

    # Create model
    model = Model(inputs=[x1, x2], outputs=merged)

    return model