from tensorflow.keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))

    # Main path
    x = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = Dropout(0.2)(x)
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=1, kernel_size=3, padding='same', activation='relu')(x)

    # Branch path
    y = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
    y = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(y)

    # Combine outputs
    combined = Add()([x, y])

    # Flatten and fully connected layer
    flatten = Flatten()(combined)
    outputs = Dense(units=10, activation='softmax')(flatten)

    # Model construction
    model = Model(inputs=inputs, outputs=outputs)

    return model

model = dl_model()
model.summary()