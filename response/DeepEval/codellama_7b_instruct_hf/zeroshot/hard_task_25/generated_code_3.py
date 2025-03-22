from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, UpSampling2D, Concatenate, Add
from keras.applications import VGG16

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    x = Conv2D(32, (1, 1), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)

    # Branch path
    branch_input = Conv2D(64, (1, 1), activation='relu')(x)
    branch_1 = Conv2D(32, (3, 3), activation='relu')(branch_input)
    branch_2 = MaxPooling2D((2, 2))(branch_input)
    branch_2 = Conv2D(32, (3, 3), activation='relu')(branch_2)
    branch_3 = MaxPooling2D((2, 2))(branch_input)
    branch_3 = Conv2D(32, (3, 3), activation='relu')(branch_3)

    # Concatenate branch outputs
    branch_outputs = Concatenate()([branch_1, branch_2, branch_3])

    # Main path output
    x = Conv2D(32, (1, 1), activation='relu')(branch_outputs)
    x = GlobalAveragePooling2D()(x)

    # Branch path output
    branch_output = Conv2D(32, (1, 1), activation='relu')(branch_input)
    branch_output = GlobalAveragePooling2D()(branch_output)

    # Fuse main path and branch path outputs
    output = Add()([x, branch_output])

    # Final layer
    output = Flatten()(output)
    output = Dense(10, activation='softmax')(output)

    # Create model
    model = Model(inputs=input_layer, outputs=output)

    return model