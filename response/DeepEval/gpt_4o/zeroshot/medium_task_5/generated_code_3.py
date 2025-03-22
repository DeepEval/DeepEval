from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main Path: Two blocks of Conv2D followed by MaxPooling2D
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Branch Path: One block of Conv2D followed by MaxPooling2D
    y = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    y = MaxPooling2D(pool_size=(2, 2))(y)

    # Combine outputs from main and branch paths
    combined = Add()([x, y])

    # Flatten the combined output
    flat = Flatten()(combined)

    # Fully connected layers
    fc1 = Dense(128, activation='relu')(flat)
    output_layer = Dense(10, activation='softmax')(fc1)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# To instantiate the model, you would call the function like this:
# model = dl_model()
# model.summary()  # To print the model's architecture