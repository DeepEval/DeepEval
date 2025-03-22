import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Add

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Path 1
    x1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    x1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x1)
    x1 = AveragePooling2D(pool_size=(2, 2))(x1)
    x1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x1)
    x1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x1)
    x1 = AveragePooling2D(pool_size=(2, 2))(x1)

    # Path 2
    x2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    x2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x2)
    x2 = AveragePooling2D(pool_size=(2, 2))(x2)

    # Add the outputs of both paths
    combined = Add()([x1, x2])

    # Flatten the combined output
    flatten = Flatten()(combined)

    # Fully connected layer
    output_layer = Dense(10, activation='softmax')(flatten)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()