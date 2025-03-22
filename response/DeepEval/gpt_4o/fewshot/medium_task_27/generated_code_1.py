import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # First branch with 3x3 convolutions
    conv_branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second branch with 5x5 convolutions
    conv_branch2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine features from both branches
    combined_features = Add()([conv_branch1, conv_branch2])

    # Global Average Pooling
    global_avg_pooling = GlobalAveragePooling2D()(combined_features)

    # Attention mechanism
    attention_dense1 = Dense(units=32, activation='relu')(global_avg_pooling)
    attention_weights1 = Dense(units=32, activation='softmax')(attention_dense1)
    attention_weights2 = Dense(units=32, activation='softmax')(attention_dense1)

    # Weighted features
    weighted_branch1 = Multiply()([conv_branch1, attention_weights1])
    weighted_branch2 = Multiply()([conv_branch2, attention_weights2])

    # Add weighted features
    weighted_combined = Add()([weighted_branch1, weighted_branch2])

    # Flatten layer to transition to fully connected layers
    global_avg_pooling_final = GlobalAveragePooling2D()(weighted_combined)

    # Fully connected output layer
    output_layer = Dense(units=10, activation='softmax')(global_avg_pooling_final)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model