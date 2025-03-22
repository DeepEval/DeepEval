from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, concatenate, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Main path
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Branch path
    branch_x = GlobalAveragePooling2D()(inputs)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    
    # Generate channel weights
    weights = Dense(2)(branch_x)
    weights = Lambda(lambda x: K.reshape(x, (-1, 2)))(weights)
    
    # Reshape and multiply with input
    x = Lambda(lambda x: K.dot(x, weights))([x])
    
    # Add outputs from both paths
    output = concatenate([x, branch_x])
    
    # Final layers
    output = Dense(10, activation='softmax')(output)  # Classification layer
    
    # Model
    model = Model(inputs=[inputs], outputs=[output])
    
    # Compile model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage:
# model = dl_model()
# model.summary()