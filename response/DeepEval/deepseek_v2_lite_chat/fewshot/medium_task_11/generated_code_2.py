import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, concatenate, Activation, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    
    # First parallel path: Global Average Pooling
    avg_pool = GlobalAveragePooling2D()(x)
    fc1 = Dense(512, activation='relu')(avg_pool)
    
    # Second parallel path: Global Max Pooling
    max_pool = GlobalMaxPooling2D()(x)
    fc2 = Dense(512, activation='relu')(max_pool)
    
    # Add the outputs of the two paths
    add_layer = Add()([fc1, fc2])
    
    # Activation function to generate channel attention weights
    attention_weights = Activation('sigmoid')(add_layer)
    
    # Element-wise multiplication with the original features
    x = keras.layers.multiply([x, attention_weights])
    
    # Average and Max Pooling to extract spatial features
    avg_pool = GlobalAveragePooling2D()(x)
    max_pool = GlobalMaxPooling2D()(x)
    
    # Concatenate along the channel dimension
    concat = concatenate([avg_pool, max_pool])
    
    # Fully connected layer for final classification
    output = Dense(10, activation='softmax')(concat)
    
    # Model
    model = keras.Model(inputs=inputs, outputs=output)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])