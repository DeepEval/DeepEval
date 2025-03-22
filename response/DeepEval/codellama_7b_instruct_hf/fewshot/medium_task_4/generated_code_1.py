import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first pathway
    path1 = Conv2D(32, (3, 3), activation='relu')(input_shape)
    path1 = MaxPooling2D((2, 2))(path1)
    path1 = Conv2D(64, (3, 3), activation='relu')(path1)
    path1 = MaxPooling2D((2, 2))(path1)
    path1 = Flatten()(path1)

    # Define the second pathway
    path2 = Conv2D(32, (3, 3), activation='relu')(input_shape)
    path2 = Flatten()(path2)

    # Add the outputs from both pathways
    merged = keras.layers.Add()([path1, path2])

    # Map the merged output to a probability distribution over the 10 classes
    output = Dense(10, activation='softmax')(merged)

    # Define the model
    model = keras.models.Model(input_shape, output)

    return model