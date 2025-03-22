import keras
from keras import layers
from keras.applications import ResNet

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first level of the model
    first_level = ResNet(
        input_shape=input_shape,
        blocks=(
            layers.Conv2D(16, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Flatten(),
        ),
        output_activation='sigmoid'
    )

    # Define the second level of the model
    second_level = ResNet(
        input_shape=input_shape,
        blocks=(
            layers.Conv2D(16, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Flatten(),
        ),
        output_activation='sigmoid'
    )

    # Define the third level of the model
    third_level = ResNet(
        input_shape=input_shape,
        blocks=(
            layers.Conv2D(16, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Flatten(),
        ),
        output_activation='sigmoid'
    )

    # Define the model
    model = keras.Sequential(
        [
            first_level,
            layers.Add()([first_level, second_level]),
            layers.Add()([second_level, third_level]),
            layers.GlobalAveragePooling2D(),
            layers.Dense(10, activation='softmax')
        ]
    )

    return model