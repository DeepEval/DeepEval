import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def basic_block(x, filters, kernel_size):
    # Main path
    main_path = Conv2D(filters, kernel_size, padding='same')(x)
    main_path = BatchNormalization()(main_path)
    main_path = ReLU()(main_path)

    # Branch path (identity)
    branch_path = Conv2D(filters, kernel_size, padding='same')(x)
    branch_path = BatchNormalization()(branch_path)

    # Feature fusion
    output = Add()([main_path, branch_path])
    output = ReLU()(output)
    
    return output

def dl_model():
    inputs = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    x = Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
    
    # First basic block
    x = basic_block(x, 16, (3, 3))
    
    # Second basic block
    x = basic_block(x, 16, (3, 3))
    
    # Average pooling and flattening
    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)
    
    # Fully connected layer
    outputs = Dense(10, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()