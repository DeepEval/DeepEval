import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, Concatenate

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First feature extraction path (1x1 convolution)
    path1 = Conv2D(32, (1, 1), activation='relu')(inputs)
    path1 = MaxPooling2D((2, 2))(path1)

    # Second feature extraction path (sequence of convolutions)
    path2 = Conv2D(32, (1, 1), activation='relu')(inputs)
    path2 = Conv2D(32, (1, 7), activation='relu')(path2)
    path2 = Conv2D(32, (7, 1), activation='relu')(path2)
    path2 = MaxPooling2D((2, 2))(path2)

    # Concatenate the outputs of the two paths
    concatenated = Concatenate()([path1, path2])

    # 1x1 convolution to align the output dimensions with the input image's channel
    main_path = Conv2D(64, (1, 1), activation='relu')(concatenated)

    # Branch connecting directly to the input
    branch = Conv2D(64, (1, 1), activation='relu')(inputs)
    branch = MaxPooling2D((2, 2))(branch)

    # Merge the outputs of the main path and the branch through addition
    merged = Add()([main_path, branch])

    # Flatten the output
    flattened = Flatten()(merged)

    # Two fully connected layers for classification
    output = Dense(128, activation='relu')(flattened)
    output = Dense(10, activation='softmax')(output)

    # Create the model
    model = Model(inputs=inputs, outputs=output)

    return model

# Create the model
model = dl_model()

# Display the model summary
model.summary()