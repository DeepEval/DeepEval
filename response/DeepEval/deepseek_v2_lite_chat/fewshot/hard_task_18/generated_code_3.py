import keras
from keras.models import Model
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First block
    def block_1(x):
        # Convolutional layers
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        # Average pooling
        avg_pool = AveragePooling2D(pool_size=(2, 2))(conv2)
        # Add block_1 output and input to main path
        add_layer = Add()([avg_pool, x])
        return add_layer
    
    # Second block
    def block_2(x):
        # Global average pooling
        avg_pool = GlobalAveragePooling2D()(x)
        # Two fully connected layers
        dense1 = Dense(units=128, activation='relu')(avg_pool)
        dense2 = Dense(units=64, activation='relu')(dense1)
        # Refine channel weights
        channel_weights = Dense(units=x.shape[-1])(dense2)
        # Multiply channel weights with input
        refined_input = keras.layers.Multiply()([x, channel_weights])
        # Flatten and pass through final fully connected layer
        flat_layer = Flatten()(refined_input)
        output_layer = Dense(units=10, activation='softmax')(flat_layer)
        return output_layer
    
    # Build the model
    model = Model(inputs=input_layer, outputs=block_2(block_1(input_layer)))
    
    return model

# Create and return the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])