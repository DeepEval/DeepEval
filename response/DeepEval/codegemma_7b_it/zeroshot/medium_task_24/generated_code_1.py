from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for CIFAR-10 images
    inputs = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution, 3x3 convolution
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(inputs)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch1)

    # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(inputs)
    branch2 = Conv2D(filters=64, kernel_size=(1, 7), activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(7, 1), activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)

    # Branch 3: Max pooling
    branch3 = MaxPooling2D(pool_size=(2, 2))(inputs)

    # Concatenate outputs from all branches
    concat_layer = concatenate([branch1, branch2, branch3])

    # Dropout layer
    dropout_layer = Dropout(rate=0.5)(concat_layer)

    # Fully connected layers
    flatten_layer = Flatten()(dropout_layer)
    dense_layer1 = Dense(units=128, activation='relu')(flatten_layer)
    dense_layer2 = Dense(units=64, activation='relu')(dense_layer1)
    dense_layer3 = Dense(units=10, activation='softmax')(dense_layer2)

    # Create the model
    model = Model(inputs=inputs, outputs=dense_layer3)

    return model