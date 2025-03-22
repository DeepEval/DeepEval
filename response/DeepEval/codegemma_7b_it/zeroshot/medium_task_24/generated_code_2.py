from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.layers import concatenate

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Branch 1
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch1)

    # Branch 2
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)
    branch2 = Conv2D(filters=16, kernel_size=(1, 7), activation='relu')(branch2)
    branch2 = Conv2D(filters=16, kernel_size=(7, 1), activation='relu')(branch2)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch2)

    # Branch 3
    branch3 = MaxPooling2D(pool_size=(2, 2))(inputs)

    # Concatenate outputs from all branches
    concat = concatenate([branch1, branch2, branch3])

    # Dropout
    concat = Dropout(0.25)(concat)

    # Flatten
    flatten = Flatten()(concat)

    # Fully connected layers
    fc1 = Dense(512, activation='relu')(flatten)
    fc2 = Dense(256, activation='relu')(fc1)
    fc3 = Dense(10, activation='softmax')(fc2)

    # Model definition
    model = Model(inputs=inputs, outputs=fc3)

    return model