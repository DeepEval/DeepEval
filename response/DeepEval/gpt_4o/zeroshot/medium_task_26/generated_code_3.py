from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (32, 32, 64)
    inputs = Input(shape=input_shape)

    # Step 1: Compressing the input channels with a 1x1 convolutional layer
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)

    # Step 2: Expanding features with parallel convolutional layers
    # 1x1 convolution
    conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
    # 3x3 convolution
    conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)

    # Step 3: Concatenate the results
    concatenated = Concatenate()([conv1x1, conv3x3])

    # Step 4: Flatten the output feature map
    flattened = Flatten()(concatenated)

    # Step 5: Fully connected layers
    fc1 = Dense(units=128, activation='relu')(flattened)
    output = Dense(units=10, activation='softmax')(fc1)  # Assuming 10 classes for classification

    # Construct the model
    model = Model(inputs=inputs, outputs=output)

    return model

# Example of creating the model
model = dl_model()
model.summary()