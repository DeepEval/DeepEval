from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    inputs = Input(shape=(32, 32, 3))

    # First Max Pooling Layer
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(x)
    x = Flatten()(x)

    # Second Max Pooling Layer
    y = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(inputs)
    y = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(y)
    y = Flatten()(y)

    # Third Max Pooling Layer
    z = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(inputs)
    z = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(z)
    z = Flatten()(z)

    # Concatenation Layer
    concat = concatenate([x, y, z])

    # Fully Connected Layers
    dense1 = Dense(units=64, activation='relu')(concat)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Model Creation
    model = Model(inputs=inputs, outputs=dense2)

    return model

# Print the model summary
model = dl_model()
model.summary()