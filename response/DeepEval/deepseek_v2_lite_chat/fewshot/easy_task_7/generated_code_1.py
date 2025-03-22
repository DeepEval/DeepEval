import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Dropout

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    def main_path(x):
        # Convolutional layer with ReLU activation
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        # Dropout to prevent overfitting
        conv1 = Dropout(0.5)(conv1)
        # Max pooling layer
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        # Second convolutional layer
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
        # Dropout to prevent overfitting
        conv2 = Dropout(0.5)(conv2)
        # Flatten layer
        flatten = Flatten()(conv2)
        return flatten
    
    # Branch path
    def branch_path(x):
        # Directly pass through without any transformations
        branch = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        return branch
    
    # Main path output
    main_output = main_path(input_layer)
    # Branch path output
    branch_output = branch_path(input_layer)
    
    # Combine outputs from both paths
    combined = Add()([main_output, branch_output])
    
    # Flattening layer
    flatten = Flatten()(combined)
    
    # Fully connected layer
    dense = Dense(units=128, activation='relu')(flatten)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Construct the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])