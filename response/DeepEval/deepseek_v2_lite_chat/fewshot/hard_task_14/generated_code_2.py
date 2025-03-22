import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D

def dl_model():

    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Main path
    x = inputs
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dense(units=32, activation='relu')(x)
    x = Dense(units=10, activation='softmax')(x)

    # Branch path
    y = inputs
    y = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(y)
    y = MaxPooling2D()(y)
    y = GlobalAveragePooling2D()(y)
    y = Dense(units=128, activation='relu')(y)
    y = Dense(units=64, activation='relu')(y)
    y = Dense(units=32, activation='relu')(y)
    y = Dense(units=10, activation='softmax')(y)

    # Combine paths
    combined = keras.layers.concatenate([x, y])
    output = Dense(units=10, activation='softmax')(combined)

    model = keras.Model(inputs=[inputs, y], outputs=output)

    return model

model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])