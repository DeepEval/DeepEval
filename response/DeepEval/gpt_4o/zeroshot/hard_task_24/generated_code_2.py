from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for CIFAR-10 images (32x32 RGB images)
    input_layer = Input(shape=(32, 32, 3))

    # Initial 1x1 convolutional layer
    x = Conv2D(32, (1, 1), activation='relu')(input_layer)

    # Branch 1: Local feature extraction with a 3x3 convolutional layer
    branch1 = Conv2D(32, (3, 3), padding='same', activation='relu')(x)

    # Branch 2: Max pooling, 3x3 convolutional layer, upsampling
    branch2 = MaxPooling2D(pool_size=(2, 2))(x)
    branch2 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)

    # Branch 3: Max pooling, 3x3 convolutional layer, upsampling
    branch3 = MaxPooling2D(pool_size=(2, 2))(x)
    branch3 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)

    # Concatenate the outputs of all branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # 1x1 convolutional layer after concatenation
    x = Conv2D(32, (1, 1), activation='relu')(concatenated)

    # Flatten the feature maps
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)  # Output layer for 10-class classification

    # Create the model
    model = Model(inputs=input_layer, outputs=x)

    return model