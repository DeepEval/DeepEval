from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the first specialized block
    block1 = Conv2D(32, (3, 3), activation='relu')(Input(shape=input_shape))
    block1 = Conv2D(32, (1, 1), activation='relu')(block1)
    block1 = MaxPooling2D((2, 2))(block1)
    block1 = Dropout(0.2)(block1)

    # Define the second specialized block
    block2 = Conv2D(64, (3, 3), activation='relu')(block1)
    block2 = Conv2D(64, (1, 1), activation='relu')(block2)
    block2 = MaxPooling2D((2, 2))(block2)
    block2 = Dropout(0.2)(block2)

    # Define the global average pooling layer
    block2 = GlobalAveragePooling2D()(block2)

    # Define the flattening layer
    flat = Flatten()(block2)

    # Define the fully connected layer
    dense = Dense(10, activation='softmax')(flat)

    # Create the model
    model = Model(inputs=Input(shape=input_shape), outputs=dense)

    return model