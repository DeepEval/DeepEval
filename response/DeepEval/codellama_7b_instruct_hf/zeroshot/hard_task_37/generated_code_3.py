from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Add, Concatenate
from keras.applications.vgg16 import VGG16

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the first parallel branch
    branch1 = Conv2D(32, (3, 3), activation='relu')(input_shape)
    branch1 = Conv2D(64, (3, 3), activation='relu')(branch1)
    branch1 = Conv2D(64, (3, 3), activation='relu')(branch1)
    branch1 = MaxPooling2D((2, 2))(branch1)

    # Define the second parallel branch
    branch2 = Conv2D(32, (3, 3), activation='relu')(input_shape)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)
    branch2 = MaxPooling2D((2, 2))(branch2)

    # Define the concatenation layer
    concat_layer = Concatenate()([branch1, branch2])

    # Define the flattening layer
    flatten_layer = Flatten()(concat_layer)

    # Define the fully connected layer
    fc_layer = Dense(128, activation='relu')(flatten_layer)

    # Define the output layer
    output_layer = Dense(10, activation='softmax')(fc_layer)

    # Create the model
    model = Model(inputs=input_shape, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model