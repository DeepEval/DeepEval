import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path: Global average pooling and dense layers to generate weights
    global_avg_pooling = GlobalAveragePooling2D()(input_layer)
    dense_main1 = Dense(units=64, activation='relu')(global_avg_pooling)  # Adjust the units as needed
    dense_main2 = Dense(units=3, activation='sigmoid')(dense_main1)  # To match the channel number of input layer
    weights_reshaped = keras.layers.Reshape((1, 1, 3))(dense_main2)  # Reshape to match input layer shape

    # Element-wise multiplication with input feature map
    scaled_features = Multiply()([input_layer, weights_reshaped])

    # Branch path: Directly connected to the input layer
    branch_path = input_layer  # No modification

    # Combine main and branch paths
    combined_features = Add()([scaled_features, branch_path])

    # Final fully connected layers
    flatten_combined = Flatten()(combined_features)
    dense_final1 = Dense(units=128, activation='relu')(flatten_combined)
    output_layer = Dense(units=10, activation='softmax')(dense_final1)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()