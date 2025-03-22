from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, concatenate

def dl_model():
    inputs = Input(shape=(32, 32, 3))

    # Branch 1
    branch_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)
    branch_1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch_1)

    # Branch 2
    branch_2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)
    branch_2 = Conv2D(filters=16, kernel_size=(1, 7), activation='relu')(branch_2)
    branch_2 = Conv2D(filters=16, kernel_size=(7, 1), activation='relu')(branch_2)
    branch_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch_2)

    # Branch 3
    branch_3 = MaxPooling2D(pool_size=(2, 2))(inputs)

    # Concatenate branches
    merged = concatenate([branch_1, branch_2, branch_3])

    # Fully connected layers
    x = Flatten()(merged)
    x = Dense(units=100, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=50, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    outputs = Dense(units=10, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model