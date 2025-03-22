import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    # Three parallel paths with different pooling sizes
    path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)

    # Flatten and dropout regularization
    flat1 = Flatten()(path1)
    flat2 = Flatten()(path2)
    flat3 = Flatten()(path3)
    dropout1 = Dropout(0.2)(flat1)
    dropout2 = Dropout(0.2)(flat2)
    dropout3 = Dropout(0.2)(flat3)

    # Concatenate outputs
    concatenated = concatenate([flat1, flat2, flat3])

    # Block 2
    # Four parallel paths with different convolution and pooling strategies
    path1 = Conv2D(64, (1, 1), activation='relu')(concatenated)
    path2 = Conv2D(64, (1, 7), activation='relu')(concatenated)
    path3 = Conv2D(64, (7, 1), activation='relu')(concatenated)
    path4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(concatenated)

    # Flatten and dropout regularization
    flat1 = Flatten()(path1)
    flat2 = Flatten()(path2)
    flat3 = Flatten()(path3)
    flat4 = Flatten()(path4)
    dropout1 = Dropout(0.2)(flat1)
    dropout2 = Dropout(0.2)(flat2)
    dropout3 = Dropout(0.2)(flat3)
    dropout4 = Dropout(0.2)(flat4)

    # Concatenate outputs
    concatenated = concatenate([flat1, flat2, flat3, flat4])

    # Final classification
    output = Dense(10, activation='softmax')(concatenated)

    # Create and return the model
    model = Model(inputs=input_layer, outputs=output)
    return model