from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Flatten, BatchNormalization, Activation, Add, Concatenate
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    # Path 1: Global average pooling followed by two fully connected layers
    path1 = GlobalAveragePooling2D()(input_layer)
    path1 = Dense(64, activation='relu')(path1)
    path1 = Dense(128, activation='relu')(path1)

    # Path 2: Global max pooling followed by two fully connected layers
    path2 = GlobalMaxPooling2D()(input_layer)
    path2 = Dense(64, activation='relu')(path2)
    path2 = Dense(128, activation='relu')(path2)

    # Concatenate the outputs from both paths
    merged_path = Concatenate()([path1, path2])

    # Channel attention weights
    attention_weights = Dense(32, activation='sigmoid')(merged_path)

    # Apply channel attention weights to the original features
    attention_output = Multiply()([merged_path, attention_weights])

    # Block 2
    # Average pooling
    avg_pooling = AveragePooling2D(pool_size=(2, 2))(attention_output)

    # Max pooling
    max_pooling = MaxPooling2D(pool_size=(2, 2))(attention_output)

    # Concatenate the outputs from both pooling layers
    concatenated_output = Concatenate()([avg_pooling, max_pooling])

    # 1x1 convolution
    conv1x1 = Conv2D(32, kernel_size=(1, 1))(concatenated_output)

    # Sigmoid activation
    activation = Activation('sigmoid')(conv1x1)

    # Add the output to the main path
    added_output = Add()([activation, attention_output])

    # Final classification
    output = Dense(10, activation='softmax')(added_output)

    # Create and return the model
    model = Model(inputs=input_layer, outputs=output)
    return model