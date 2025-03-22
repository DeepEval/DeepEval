import keras
from keras.layers import Input, Reshape, Permute, Dense, Flatten
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Step 2: Reshape the input tensor into (height, width, groups, channels_per_group)
    groups = 3
    channels_per_group = 3 // groups
    reshaped = Reshape((32, 32, groups, channels_per_group))(input_layer)

    # Step 3: Permute the tensor to shuffle the channels (swap the 3rd and 4th dimensions)
    permuted = Permute((1, 2, 4, 3))(reshaped)

    # Step 4: Reshape back to original input shape (height, width, channels)
    reshaped_back = Reshape((32, 32, 3))(permuted)

    # Step 5: Flatten the tensor for the fully connected layer
    flatten_layer = Flatten()(reshaped_back)

    # Step 6: Add a fully connected layer with softmax activation for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    # Step 7: Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# You can now create the model by calling the function
model = dl_model()
model.summary()  # Print the model summary to see the architecture