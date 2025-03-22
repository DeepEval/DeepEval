import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First part: Sequential convolution and max-pooling layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Second part: Processing through convolutional layers and dropout
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    # Third part: Upsampling with transposed convolutional layers and skip connections
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = Concatenate()([x, x_skip2])  # Assuming x_skip2 is the corresponding feature map from the first part
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = Concatenate()([x, x_skip1])  # Assuming x_skip1 is the corresponding feature map from the first part
    x = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)
    x = Concatenate()([x, input_layer])  # Adding the original input for more detailed features

    # Final 1x1 convolutional layer for classification
    x = Conv2D(10, (1, 1), activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model

# Example usage
model = dl_model()
model.summary()