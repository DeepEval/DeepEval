# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.

    The model extracts features through a combination of a 1x1 convolution and another parallel branch 
    that includes 1x1, 1x3, and 3x1 convolutions. The outputs from these two paths are concatenated 
    and passed through another 1x1 convolution to produce the main output with the same dimensions 
    as the channel of input. Additionally, a direct connection from the input to the model's branch 
    allows for fusion with the main pathway via an additive operation. Finally, the classification 
    probabilities are generated through two fully connected layers.
    """

    # Define the input shape based on the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Create the base model with the main pathway
    x = layers.Input(shape=input_shape)
    pathway1 = layers.Conv2D(32, 1, activation='relu')(x)  # 1x1 convolution
    pathway2 = layers.Concatenate()([
        layers.Conv2D(32, 1, activation='relu')(x),  # 1x1 convolution
        layers.Conv2D(32, 3, padding='same', activation='relu')(pathway2),  # 1x3 convolution
        layers.Conv2D(32, 1, padding='same', activation='relu')(pathway2)  # 3x1 convolution
    ])
    pathway = layers.Concatenate()([pathway1, pathway2])
    main_output = layers.Conv2D(3, 1, activation='relu')(pathway)  # 1x1 convolution to produce the main output

    # Add a direct connection from the input to the model's branch
    branch_input = layers.Concatenate()([x, main_output])

    # Create the model by defining the output of the additive operation and the classification probabilities
    model = keras.Model(inputs=x, outputs=layers.Add()([branch_input, main_output]))
    model = layers.GlobalAveragePooling2D()(model)
    model = layers.Dense(64, activation='relu')(model)
    model = layers.Dense(10, activation='softmax')(model)

    return model

model = dl_model()
model.summary()