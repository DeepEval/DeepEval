from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Conv2DTranspose
from keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First part: extract deep features
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Second part: enhance generalization capabilities
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(1024, (3, 3), activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(2048, (3, 3), activation='relu')(x)
    x = Dropout(0.5)(x)

    # Third part: upsample and restore spatial information
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(1024, (3, 3), activation='relu')(x)
    x = Conv2DTranspose(512, (3, 3), activation='relu')(x)
    x = Conv2DTranspose(256, (3, 3), activation='relu')(x)
    x = Conv2DTranspose(128, (3, 3), activation='relu')(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu')(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu')(x)

    # Output layer
    outputs = Conv2D(10, (1, 1), activation='softmax')(x)

    # Create and compile the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model