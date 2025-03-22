from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.applications.vgg16 import VGG16

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the main path
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Define the branch path
    branch_input = Input(shape=(32, 32, 3))
    branch = Conv2D(32, (5, 5), activation='relu')(branch_input)

    # Combine the main and branch paths
    x = Concatenate()([x, branch])

    # Flatten the output of the concatenation
    x = Flatten()(x)

    # Add the fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=[input_layer, branch_input], outputs=x)

    return model


from keras.applications import VGG16

def dl_model():
    # Load the VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add a new top layer
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=base_model.input, outputs=x)

    return model