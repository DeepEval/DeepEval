from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, concatenate, Dense, Reshape
from keras.layers import Conv2D, MaxPooling2D, Flatten, concatenate, Dense, Reshape

def dl_model():
    # Define the first block
    input_block = Input(shape=(28, 28, 1))  # Input shape should be (28, 28, 1) for grayscale images
    
    # Define parallel paths with average pooling layers
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_block)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_block)
    pool4 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_block)
    
    # Flatten the results
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat4 = Flatten()(pool4)
    
    # Concatenate the flattened outputs
    concat = concatenate([flat1, flat2, flat4])
    
    # First fully connected layer
    dense1 = Dense(512, activation='relu')(concat)
    
    # Reshape the output for the second block
    reshape = Reshape((-1, 16))(dense1)  # Assuming 16 features after concatenation
    
    # Define the second block
    block2 = Input(shape=(16,))
    
    # Process through various configurations
    conv1x1 = Conv2D(32, kernel_size=(1, 1), activation='relu')(block2)
    conv3x3 = Conv2D(32, kernel_size=(3, 3), activation='relu')(block2)
    
    conv1x7_1x1 = Conv2D(32, kernel_size=(1, 7), activation='relu')(block2)
    conv7x1_1x7 = Conv2D(32, kernel_size=(7, 1), activation='relu')(block2)
    
    avg_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(block2)
    
    # Concatenate the outputs from the different configurations
    concat2 = concatenate([conv1x1, conv3x3, conv1x7_1x1, conv7x1_1x7, avg_pool])
    
    # Another fully connected layer
    dense2 = Dense(128, activation='relu')(concat2)
    
    # Output layer
    output = Dense(10, activation='softmax')(dense2)  # Assuming 10 classes
    
    # Create the model
    model = Model(inputs=[input_block, block2], outputs=output)
    
    return model

# Example usage:
x_train, x_test, y_train, y_test = load_data()
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([x_train, y_train], y_train, epochs=10, batch_size=64, validation_split=0.2)