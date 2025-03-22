import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, Flatten
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    # Step 2: Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(input_layer)

    # Step 3: Two fully connected layers to generate weights
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense2 = Dense(units=32 * 32 * 3, activation='sigmoid')(dense1)  # 32*32*3 to match the input shape for reshaping

    # Step 4: Reshape to match input layer shape
    reshaped_weights = Reshape((32, 32, 3))(dense2)

    # Step 5: Multiply element-wise with the input feature map
    weighted_input = Multiply()([input_layer, reshaped_weights])

    # Branch path
    # This path is just the input layer without modification
    branch_path = input_layer

    # Step 6: Combine main and branch paths
    combined_output = Add()([weighted_input, branch_path])

    # Step 7: Feed through two fully connected layers
    flat_combined_output = Flatten()(combined_output)
    final_dense1 = Dense(units=128, activation='relu')(flat_combined_output)
    final_output = Dense(units=10, activation='softmax')(final_dense1)

    # Create model
    model = Model(inputs=input_layer, outputs=final_output)

    return model

# Example usage
model = dl_model()
model.summary()