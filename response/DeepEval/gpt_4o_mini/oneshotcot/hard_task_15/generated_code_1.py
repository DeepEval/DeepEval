import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Add
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))

    # Step 2: Main path
    # Global Average Pooling to extract global information
    pooled_features = GlobalAveragePooling2D()(input_layer)
    
    # Two fully connected layers
    dense1 = Dense(units=128, activation='relu')(pooled_features)
    dense2 = Dense(units=3 * 32 * 32, activation='relu')(dense1)  # Producing weights for 3 channels and 32x32 feature map
    
    # Reshape to match the input layer's shape
    reshaped_weights = Dense(units=3 * 32 * 32, activation='sigmoid')(dense2)  # Sigmoid for weights between 0 and 1
    reshaped_weights = keras.layers.Reshape((32, 32, 3))(reshaped_weights)
    
    # Element-wise multiplication with the input feature map
    main_output = Multiply()([input_layer, reshaped_weights])

    # Step 3: Branch path
    # Directly connected to the input layer without modification
    branch_output = input_layer

    # Step 4: Combine main and branch paths
    combined_output = Add()([main_output, branch_output])

    # Step 5: Feed through two fully connected layers for final output
    combined_flatten = GlobalAveragePooling2D()(combined_output)  # Flattening to vector for Dense layers
    final_dense1 = Dense(units=128, activation='relu')(combined_flatten)
    final_output = Dense(units=10, activation='softmax')(final_dense1)  # 10 classes for CIFAR-10

    # Step 6: Build the model
    model = Model(inputs=input_layer, outputs=final_output)

    return model