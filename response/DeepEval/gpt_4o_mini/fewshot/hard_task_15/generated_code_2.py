import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main Path
    pooled_features = GlobalAveragePooling2D()(input_layer)
    fc1 = Dense(units=128, activation='relu')(pooled_features)
    fc2 = Dense(units=32 * 32 * 3, activation='sigmoid')(fc1)  # Output shape matches input shape
    reshaped_weights = Dense(units=32 * 32 * 3, activation='sigmoid')(fc2)  # Matching the number of channels
    reshaped_weights = keras.layers.Reshape((32, 32, 3))(reshaped_weights)
    main_path_output = Multiply()([input_layer, reshaped_weights])  # Element-wise multiplication

    # Branch Path
    branch_path_output = input_layer  # No modification, just pass through

    # Combine both paths
    combined_output = Add()([main_path_output, branch_path_output])

    # Fully Connected Layers for final classification
    flattened = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    model = Model(inputs=input_layer, outputs=output_layer)

    return model