# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the main path
    main_input = Input(shape=input_shape)
    main_conv1 = Conv2D(32, (3, 3), activation='relu')(main_input)
    main_conv1 = Conv2D(32, (3, 3), activation='relu')(main_conv1)
    main_pool1 = MaxPooling2D((2, 2))(main_conv1)
    main_conv2 = Conv2D(64, (3, 3), activation='relu')(main_pool1)
    main_conv2 = Conv2D(64, (3, 3), activation='relu')(main_conv2)
    main_pool2 = MaxPooling2D((2, 2))(main_conv2)

    # Define the branch path
    branch_input = Input(shape=input_shape)
    branch_conv = Conv2D(64, (3, 3), activation='relu')(branch_input)
    branch_conv = Conv2D(64, (3, 3), activation='relu')(branch_conv)
    branch_pool = MaxPooling2D((2, 2))(branch_conv)

    # Combine the main path and branch path using addition
    combined = Add()([main_pool2, branch_pool])

    # Add a flattening layer to the combined output
    flattened = Flatten()(combined)

    # Add two fully connected layers for classification
    fc1 = Dense(64, activation='relu')(flattened)
    fc2 = Dense(10, activation='softmax')(fc1)

    # Define the model
    model = Model(inputs=[main_input, branch_input], outputs=fc2)

    return model

# Create and return the model
model = dl_model()
print(model.summary())