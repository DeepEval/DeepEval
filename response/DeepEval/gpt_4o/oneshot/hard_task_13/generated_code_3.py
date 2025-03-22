import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Multiply, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Four parallel branches
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    
    block1_output = Concatenate()([path1, path2, path3, path4])

    # Block 2: Dimensionality reduction and attention mechanism
    global_avg_pool = GlobalAveragePooling2D()(block1_output)
    
    # Fully connected layers to generate channel weights
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense2 = Dense(units=block1_output.shape[-1], activation='sigmoid')(dense1)
    
    # Reshape and apply the weights
    reshaped_weights = Reshape((1, 1, block1_output.shape[-1]))(dense2)
    scaled_features = Multiply()([block1_output, reshaped_weights])
    
    # Final output layer
    flatten_layer = GlobalAveragePooling2D()(scaled_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()