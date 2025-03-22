import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model

def dl_model():
    # First block: Extract features
    input_layer = Input(shape=(32, 32, 3))  # Assuming input images are 32x32
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = AveragePooling2D()(x)
    
    # Main path: Refine features
    main_output = x  # Output of the first block
    
    # Second block: Generate weights
    z = GlobalAveragePooling2D()(main_output)  # Global average pooling to generate channel weights
    z = Dense(128, activation='relu')(z)  # Two fully connected layers
    weights = Dense(main_output.shape[-1], activation='sigmoid')(z)  # Refined weights
    
    # Concatenate the main path output with the refined weights
    combined = keras.layers.concatenate([main_output, weights])
    
    # Final classification layer
    output = Dense(10, activation='softmax')(combined)  # Assuming 10 classes
    
    # Build the model
    model = Model(inputs=[input_layer, weights], outputs=output)
    
    return model

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])