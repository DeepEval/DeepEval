import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Input, Add
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Branch path
    branch_layer = GlobalAveragePooling2D()(x)
    branch_layer = Dense(128, activation='relu')(branch_layer)
    branch_layer = Dense(10, activation='softmax')(branch_layer)

    # Output layer
    output_layer = Add()([x, branch_layer])
    output_layer = Flatten()(output_layer)
    output_layer = Dense(10, activation='softmax')(output_layer)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model