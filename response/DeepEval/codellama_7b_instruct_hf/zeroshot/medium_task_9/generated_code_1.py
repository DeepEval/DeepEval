import keras
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Add, Activation, AveragePooling2D, Flatten, Dense
from keras.applications.vgg16 import VGG16

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    x = Conv2D(16, (3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # First basic block
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    branch1 = x

    # Second basic block
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    branch2 = x

    # Feature fusion
    x = Add()([branch1, branch2])
    x = Activation('relu')(x)

    # Average pooling layer
    x = AveragePooling2D((2, 2))(x)

    # Flatten and fully connected layers
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Create and compile the model
    model = Model(inputs=input_layer, outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model