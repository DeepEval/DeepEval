import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Second convolutional layer with concatenation
    concat1 = Concatenate(axis=-1)([input_layer, conv1])
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(concat1)
    
    # Third convolutional layer with concatenation
    concat2 = Concatenate(axis=-1)([concat1, conv2])
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(concat2)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(conv3)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()