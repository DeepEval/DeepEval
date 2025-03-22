import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Lambda

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        return pool2
    
    # Branch path
    def branch_path(input_tensor):
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        return conv3
    
    # Combine main and branch paths
    combined_tensor = Add()([main_path(input_tensor), branch_path(input_tensor)])
    
    # Flatten and pass through fully connected layers
    flattened = Flatten()(combined_tensor)
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])