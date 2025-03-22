import keras
from keras.layers import Input, Conv2D, MaxPooling2D, ReLU, SeparableConv2D, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    def block1(input_tensor):
        # ReLU activation
        relu1 = ReLU()(input_tensor)
        # Separable convolutional layer
        sep_conv1 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same')(relu1)
        # Max pooling
        maxpool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(sep_conv1)
        return maxpool1
    
    def block2(input_tensor):
        # Max pooling
        maxpool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(input_tensor)
        return maxpool2
    
    # Branch path
    def branch_block(input_tensor):
        # 1x1 convolutional layer
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_tensor)
        return conv1
    
    # Combine main and branch paths
    main_path = block1(input_layer)
    branch_path = branch_block(input_tensor=input_layer)
    merged_path = Add()([main_path, branch_path])
    
    # Flatten and fully connected layers
    flattened = Flatten()(merged_path)
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Model architecture
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])