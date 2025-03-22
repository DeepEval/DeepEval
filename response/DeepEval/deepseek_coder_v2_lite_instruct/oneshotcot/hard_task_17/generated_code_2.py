import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        # Global Average Pooling
        gap = GlobalAveragePooling2D()(input_tensor)
        # Two fully connected layers with the same channel as the input layer
        fc1 = Dense(units=32, activation='relu')(gap)
        fc2 = Dense(units=32, activation='relu')(fc1)
        # Reshape weights to match the input shape
        weights = Dense(units=32*32*3, activation='sigmoid')(fc2)
        # Multiply weights with the input to produce the weighted feature output
        weighted_features = Multiply()([input_tensor, weights])
        return weighted_features

    block1_output = block1(input_layer)

    # Block 2
    def block2(input_tensor):
        # Two 3x3 convolutional layers
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1)
        # Max Pooling
        max_pooling = MaxPooling2D(pool_size=(2, 2))(conv2)
        return max_pooling

    block2_output = block2(block1_output)

    # Branch from Block 1
    branch_output = block1(block2_output)

    # Add the main path and the branch outputs
    fused_output = Add()([block2_output, branch_output])

    # Flatten the output
    flattened_output = Flatten()(fused_output)

    # Two fully connected layers for classification
    dense1 = Dense(units=64, activation='relu')(flattened_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()
model.summary()