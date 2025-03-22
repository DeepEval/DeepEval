import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, Lambda

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Main path
    x = Conv2D(32, (1, 1), activation='relu')(inputs)
    main_path_1x1 = Conv2D(32, (1, 1), activation='relu')(inputs)
    main_path_3x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    main_path_5x5 = Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
    
    # Split the main path outputs
    split_1x1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(main_path_1x1)
    split_3x3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(main_path_3x3)
    split_5x5 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(main_path_5x5)
    
    # Concatenate the split outputs
    concatenated = Add()([split_1x1[0], split_3x3[1], split_5x5[2]])

    # Branch path
    branch_path = Conv2D(32, (1, 1), activation='relu')(inputs)

    # Add the main path and branch path outputs
    fused_features = Add()([concatenated, branch_path])

    # Flatten the fused features
    flattened = Flatten()(fused_features)

    # Fully connected layers
    dense_1 = Dense(128, activation='relu')(flattened)
    outputs = Dense(10, activation='softmax')(dense_1)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()