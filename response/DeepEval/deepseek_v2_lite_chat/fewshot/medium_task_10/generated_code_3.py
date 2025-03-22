import keras
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Reduce dimensionality to 16
    x = Conv2D(filters=16, kernel_size=3, padding='same')(inputs)
    
    # Basic block
    bn1 = BatchNormalization()(x)
    relu1 = ReLU()(bn1)
    
    # Branch for feature extraction
    branch = Conv2D(filters=16, kernel_size=1, padding='same')(relu1)
    
    # Add main path and branch outputs
    add = Add()([x, branch])
    
    # First residual block
    x = Conv2D(filters=16, kernel_size=3, padding='same')(add)
    bn2 = BatchNormalization()(x)
    relu2 = ReLU()(bn2)
    # Second residual block
    x = Conv2D(filters=16, kernel_size=3, padding='same')(relu2)
    bn3 = BatchNormalization()(x)
    relu3 = ReLU()(bn3)
    # Global branch for enhanced features
    global_branch = Conv2D(filters=16, kernel_size=3, padding='same')(relu3)
    add_global = Add()([add, global_branch])
    
    # Final output
    avg_pool = AveragePooling2D(pool_size=4, strides=1)(add_global)
    flat = Flatten()(avg_pool)
    dense = Dense(units=1024, activation='relu')(flat)
    output = Dense(units=10, activation='softmax')(dense)
    
    # Create the model
    model = Model(inputs=inputs, outputs=output)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])