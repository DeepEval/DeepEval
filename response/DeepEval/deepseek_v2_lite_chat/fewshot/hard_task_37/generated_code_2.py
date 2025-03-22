import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First branch
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_path1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(branch1)
    branch1_flatten = Flatten()(main_path1)
    
    # Second branch
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(branch2)
    branch2_flatten = Flatten()(main_path2)
    
    # Parallel branch
    parallel_branch = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    parallel_branch_flatten = Flatten()(parallel_branch)
    
    # Combine outputs from both blocks
    combined_path1 = branch1_flatten + branch2_flatten
    combined_path2 = branch1_flatten + parallel_branch_flatten
    combined_output = Add()([combined_path1, combined_path2])
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(combined_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Optional: Display model summary
model.summary()