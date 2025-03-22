import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate

def dl_model():
    inputs = Input(shape=(32, 32, 3))

    # First part: Feature extraction
    x = Conv2D(64, (3, 3), activation='relu')(inputs)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Save the output of the last convolutional layer for skip connections
    conv_output = x

    # Second part: Enhancing generalization capabilities
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)

    # Third part: Upsampling with skip connections
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2))(x)
    x = concatenate([x, conv_output])
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)

    conv_output = x

    x = Conv2DTranspose(128, (2, 2), strides=(2, 2))(x)
    x = concatenate([x, conv_output])
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)

    conv_output = x

    x = Conv2DTranspose(64, (2, 2), strides=(2, 2))(x)
    x = concatenate([x, conv_output])
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)

    # Final part: Classification
    x = Conv2D(10, (1, 1), activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)

    return model

# Example usage:
# model = dl_model()
# model.summary()