from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, BatchNormalization, ReLU, Flatten, Dense
from keras.layers import LayerNormalization

def dl_model():
    # Data preparation
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Model architecture
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    x = Conv2D(32, (3, 3), padding='same', activation=ReLU())(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), padding='same', activation=ReLU())(x)
    x = BatchNormalization()(x)
    
    # Block 2
    x = Conv2D(64, (3, 3), padding='same', activation=ReLU())(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', activation=ReLU())(x)
    x = BatchNormalization()(x)
    
    # Block 3
    x = Conv2D(128, (3, 3), padding='same', activation=ReLU())(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation=ReLU())(x)
    x = BatchNormalization()(x)
    
    # Concatenate and flatten
    x = Concatenate(axis=-1)([x] * 2)
    x = Flatten()(x)
    
    # Fully connected layers
    x = Dense(1024, activation=ReLU())(x)
    x = BatchNormalization()(x)
    output_layer = Dense(10, activation='softmax')(x)  # Assuming 10 classes for CIFAR-10

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate and return the model
model = dl_model()