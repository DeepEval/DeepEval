from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, add
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    inputs = Input(shape=(28, 28, 1))

    # Main path
    main_path = Conv2D(filters=32, kernel_size=5, padding='same', activation='relu')(inputs)
    main_path = Conv2D(filters=10, kernel_size=5, padding='same', activation='relu')(main_path)

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=5, padding='same', activation='relu')(inputs)

    # Combine the main and branch paths
    combined_path = add([main_path, branch_path])

    # Flatten the output
    flatten = Flatten()(combined_path)

    # Fully connected layer for classification
    outputs = Dense(units=10, activation='softmax')(flatten)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model