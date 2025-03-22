import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, Activation, Concatenate, AveragePooling2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Parallel paths for channel attention
    def avg_pool_path(input_tensor):
        avg_pool = AveragePooling2D(pool_size=(4, 4))(input_tensor)
        flatten = Flatten()(avg_pool)
        dense1 = Dense(units=128, activation='relu')(flatten)
        dense2 = Dense(units=64, activation='relu')(dense1)
        return dense2

    avg_pool_output = avg_pool_path(input_tensor=conv1)
    
    def max_pool_path(input_tensor):
        max_pool = MaxPooling2D(pool_size=(4, 4))(input_tensor)
        flatten = Flatten()(max_pool)
        dense1 = Dense(units=128, activation='relu')(flatten)
        dense2 = Dense(units=64, activation='relu')(dense1)
        return dense2

    max_pool_output = max_pool_path(input_tensor=conv1)
    
    # Generate channel attention weights
    concat = Concatenate()([avg_pool_output, max_pool_output])
    channel_attention = Dense(units=2, activation='softmax')(concat)
    attention_weighted_features = keras.layers.multiply([avg_pool_output, channel_attention])
    
    # Spatial feature extraction
    avg_pool = AveragePooling2D(pool_size=(2, 2))(conv1)
    avg_pool_output = Flatten()(avg_pool)
    
    max_pool = MaxPooling2D(pool_size=(2, 2))(conv1)
    max_pool_output = Flatten()(max_pool)
    
    # Concatenate spatial features
    fused_features = Concatenate()([avg_pool_output, max_pool_output])
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(fused_features)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])