import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main pathway
    def main_path(input_tensor):
        # First convolutional layer
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_tensor)
        # Batch normalization
        batch_norm1 = BatchNormalization()(conv1)
        # Max pooling
        maxpool1 = MaxPooling2D(pool_size=(2, 2))(batch_norm1)
        # Second convolutional layer
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(maxpool1)
        # Batch normalization
        batch_norm2 = BatchNormalization()(conv2)
        # Max pooling
        maxpool2 = MaxPooling2D(pool_size=(2, 2))(batch_norm2)
        return maxpool2

    # Branch pathway
    def branch_path(input_tensor):
        # Single 5x5 convolutional layer
        conv = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(input_tensor)
        return conv

    # Combine features from both pathways
    main_output = main_path(input_layer)
    branch_output = branch_path(input_layer)
    combined_output = Concatenate()([main_output, branch_output])
    
    # Flatten and pass through dense layers
    flatten = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])