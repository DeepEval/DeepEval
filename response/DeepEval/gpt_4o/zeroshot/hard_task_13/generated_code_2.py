from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Concatenate, Multiply, Reshape
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 input dimensions
    num_classes = 10  # CIFAR-10 has 10 classes

    # Input layer
    inputs = Input(shape=input_shape)

    # Block 1: Parallel Branches
    # 1x1 Convolution
    branch_1x1 = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    
    # 3x3 Convolution
    branch_3x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    
    # 5x5 Convolution
    branch_5x5 = Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
    
    # 3x3 Max Pooling
    branch_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = Conv2D(32, (1, 1), activation='relu', padding='same')(branch_pool)

    # Concatenate all the branches
    concatenated = Concatenate(axis=-1)([branch_1x1, branch_3x3, branch_5x5, branch_pool])

    # Block 2: Global Average Pooling and Fully Connected Layers
    # Global Average Pooling
    gap = GlobalAveragePooling2D()(concatenated)

    # Fully Connected Layers to generate weights
    fc1 = Dense(128, activation='relu')(gap)
    fc2 = Dense(64, activation='relu')(fc1)
    fc_weights = Dense(concatenated.shape[-1], activation='sigmoid')(fc2)

    # Reshape the weights to match the feature map shape
    fc_weights_reshaped = Reshape((1, 1, concatenated.shape[-1]))(fc_weights)

    # Element-wise multiplication with the feature map
    weighted_features = Multiply()([concatenated, fc_weights_reshaped])

    # Output layer
    outputs = Dense(num_classes, activation='softmax')(GlobalAveragePooling2D()(weighted_features))

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example of creating the model
model = dl_model()
model.summary()