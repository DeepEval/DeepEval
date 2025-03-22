import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Define the input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Branch 1: 3x3 convolutions
    branch_1 = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    branch_1 = layers.Conv2D(32, (3, 3), activation='relu')(branch_1)

    # Branch 2: 1x1 convolutions followed by 3x3 convolutions
    branch_2 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch_2 = layers.Conv2D(32, (3, 3), activation='relu')(branch_2)
    branch_2 = layers.Conv2D(32, (3, 3), activation='relu')(branch_2)

    # Branch 3: Max pooling
    branch_3 = layers.MaxPooling2D((2, 2))(inputs)

    # Concatenate the outputs of the branches
    concat = layers.concatenate([branch_1, branch_2, branch_3])

    # Multi-scale feature fusion block
    fusion = layers.Conv2D(64, (1, 1), activation='relu')(concat)
    fusion = layers.BatchNormalization()(fusion)
    fusion = layers.Activation('relu')(fusion)

    # Flatten the fused feature maps
    flatten = layers.Flatten()(fusion)

    # Fully connected layers for classification
    fc = layers.Dense(512, activation='relu')(flatten)
    fc = layers.Dropout(0.5)(fc)
    outputs = layers.Dense(10, activation='softmax')(fc)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model