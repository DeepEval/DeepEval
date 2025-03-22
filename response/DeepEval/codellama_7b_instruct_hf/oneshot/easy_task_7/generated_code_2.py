from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense
from keras.models import Model

，增加dl_model 和 return model
def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Dropout(0.2)(x)

    # Branch path
    y = Conv2D(64, (3, 3), activation='relu')(input_layer)
    y = BatchNormalization()(y)
    y = Conv2D(64, (3, 3), activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Conv2D(128, (3, 3), activation='relu')(y)
    y = Dropout(0.2)(y)

    # Combine main and branch paths
    z = Add()([x, y])

    # Flatten and fully connected layers
    z = Flatten()(z)
    z = Dense(128, activation='relu')(z)
    z = Dense(10, activation='softmax')(z)

    # Create model
    model = Model(inputs=input_layer, outputs=z)

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model