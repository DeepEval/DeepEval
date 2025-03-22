import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First part: Feature extraction
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Second part: Enhance generalization capabilities
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)

    # Third part: Upsampling
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2DTranspose(64, (2, 2))(x)
    x = Conv2DTranspose(32, (2, 2))(x)
    x = Conv2DTranspose(16, (2, 2))(x)

    # Final part: 1x1 convolutional layer for output
    x = Conv2D(1, (1, 1), activation='softmax')(x)

    # Define the model
    model = Model(inputs=input_layer, outputs=x)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model