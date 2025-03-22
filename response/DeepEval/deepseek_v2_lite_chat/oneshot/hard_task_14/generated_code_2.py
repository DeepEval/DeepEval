import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    main_pool = MaxPooling2D(pool_size=(2, 2))(main_conv)
    main_flatten = Flatten()(main_pool)
    main_dense1 = Dense(units=128, activation='relu')(main_flatten)
    main_dense2 = Dense(units=64, activation='relu')(main_dense1)
    
    # Branch path
    branch_conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    branch_pool = MaxPooling2D(pool_size=(2, 2))(branch_conv)
    branch_flatten = Flatten()(branch_pool)
    
    # Combine paths
    combined_tensor = keras.layers.concatenate([main_dense2, branch_flatten])
    
    # Additional fully connected layers
    dense1 = Dense(units=128, activation='relu')(combined_tensor)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])