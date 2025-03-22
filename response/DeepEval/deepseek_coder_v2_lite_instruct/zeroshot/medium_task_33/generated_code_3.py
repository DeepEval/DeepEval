import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, SeparableConv2D, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Split the input into three channel groups
    split_layers = []
    for i in range(3):
        split_layers.append(Lambda(lambda x: x[:, :, :, i])(inputs))

    # Feature extraction for each channel group
    feature_extraction_layers = []
    for split_layer in split_layers:
        conv1x1 = SeparableConv2D(32, (1, 1), activation='relu')(split_layer)
        conv3x3 = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(split_layer)
        conv5x5 = SeparableConv2D(32, (5, 5), padding='same', activation='relu')(split_layer)
        concatenated = Concatenate()([conv1x1, conv3x3, conv5x5])
        feature_extraction_layers.append(concatenated)

    # Concatenate the feature extraction outputs
    concatenated_features = Concatenate()(feature_extraction_layers)

    # Flatten the concatenated features
    flattened = Flatten()(concatenated_features)

    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flattened)
    dense2 = Dense(128, activation='relu')(dense1)
    outputs = Dense(10, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()