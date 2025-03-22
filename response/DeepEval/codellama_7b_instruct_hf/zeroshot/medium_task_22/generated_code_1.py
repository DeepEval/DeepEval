import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Multi-scale feature fusion block
    branch1 = Conv2D(64, (3, 3), activation='relu')(input_layer)
    branch1 = Conv2D(64, (3, 3), activation='relu')(branch1)
    branch2 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)
    concat_layer = Concatenate()([branch1, branch2])
    max_pooling_layer = MaxPooling2D((2, 2))(concat_layer)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(max_pooling_layer)
    dense_layer1 = Dense(128, activation='relu')(flatten_layer)
    dense_layer2 = Dense(10, activation='softmax')(dense_layer1)

    # Create and return the model
    model = Model(inputs=input_layer, outputs=dense_layer2)
    return model