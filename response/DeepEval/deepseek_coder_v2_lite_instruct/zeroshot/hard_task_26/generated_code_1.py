import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense

def dl_model():
    # Define the input shape
    inputs = Input(shape=(32, 32, 3))

    # Main path
    x = Conv2D(32, (1, 1), activation='relu')(inputs)

    # Branch 1
    branch1 = Conv2D(64, (3, 3), activation='relu')(x)

    # Branch 2
    branch2 = MaxPooling2D((2, 2))(x)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)
    branch2 = UpSampling2D((2, 2))(branch2)

    # Branch 3
    branch3 = MaxPooling2D((2, 2))(x)
    branch3 = Conv2D(64, (3, 3), activation='relu')(branch3)
    branch3 = UpSampling2D((2, 2))(branch3)

    # Concatenate outputs from all branches
    combined = Concatenate()([branch1, branch2, branch3])

    # Final 1x1 convolutional layer
    main_output = Conv2D(64, (1, 1), activation='relu')(combined)

    # Branch path
    branch_input = Conv2D(64, (1, 1), activation='relu')(inputs)

    # Add outputs from both paths
    added = tf.add(main_output, branch_input)

    # Flatten and pass through fully connected layers
    x = Flatten()(added)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
# model = dl_model()
# model.summary()