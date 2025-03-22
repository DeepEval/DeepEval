import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Multiply, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def same_block(input_tensor):
        # Global Average Pooling
        gap = GlobalAveragePooling2D()(input_tensor)
        # Fully connected layers
        fc1 = Dense(units=128, activation='relu')(gap)
        fc2 = Dense(units=64, activation='relu')(fc1)
        # Reshape weights to match input shape
        weights = Reshape((1, 1, 64))(fc2)
        # Element-wise multiplication
        output_tensor = Multiply()([input_tensor, weights])
        return output_tensor

    # Branch 1
    branch1 = same_block(input_tensor=input_layer)
    # Branch 2
    branch2 = same_block(input_tensor=input_layer)

    # Concatenate outputs of both branches
    concatenated = Concatenate()([branch1, branch2])
    # Flatten the result
    flatten_layer = Flatten()(concatenated)
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Build and return the model
model = dl_model()
model.summary()