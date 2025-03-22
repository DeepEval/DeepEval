import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Concatenate, Flatten
from keras.datasets import cifar10

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor):
        # Global Average Pooling
        gap = GlobalAveragePooling2D()(input_tensor)
        
        # Two fully connected layers to compute weights
        fc1 = Dense(units=gap.shape[-1], activation='relu')(gap)
        fc2 = Dense(units=gap.shape[-1], activation='sigmoid')(fc1)
        
        # Reshape and multiply weights with input_tensor
        reshaped_weights = keras.layers.Reshape(target_shape=(1, 1, gap.shape[-1]))(fc2)
        scaled_features = Multiply()([input_tensor, reshaped_weights])

        return scaled_features

    # Two branches using the same block
    branch1_output = block(input_tensor=input_layer)
    branch2_output = block(input_tensor=input_layer)

    # Concatenate outputs from both branches
    concatenated_output = Concatenate()([branch1_output, branch2_output])

    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(concatenated_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Load CIFAR-10 dataset (for testing the model structure)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Construct the model
model = dl_model()

# Summary of the model
model.summary()