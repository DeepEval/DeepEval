from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, add
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_img = Input(shape=(28, 28, 1))

    # Branch 1
    branch1 = Conv2D(32, (3, 3), activation='relu')(input_img)
    branch1 = MaxPooling2D((2, 2))(branch1)
    branch1 = Conv2D(64, (3, 3), activation='relu')(branch1)
    branch1 = MaxPooling2D((2, 2))(branch1)
    branch1 = Conv2D(128, (3, 3), activation='relu')(branch1)

    # Branch 2
    branch2 = Conv2D(32, (3, 3), activation='relu')(input_img)
    branch2 = MaxPooling2D((2, 2))(branch2)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)
    branch2 = MaxPooling2D((2, 2))(branch2)
    branch2 = Conv2D(128, (3, 3), activation='relu')(branch2)

    # Parallel branch
    parallel_branch = Conv2D(128, (1, 1), activation='relu')(input_img)

    # Combine outputs from parallel branches
    combined = add([branch1, branch2, parallel_branch])

    # Flatten and fully connected layer
    flatten = Flatten()(combined)
    output = Dense(10, activation='softmax')(flatten)

    # Create the model
    model = Model(input_img, output)

    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()