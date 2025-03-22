import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, ReLU, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First level of the residual connection structure
    def residual_block(input_tensor, filters):
        # Conv layer
        conv = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(input_tensor)
        # Batch normalization
        batch_norm = BatchNormalization()(conv)
        # ReLU activation
        relu = ReLU()(batch_norm)
        
        # Branch directly connected to the block's input
        branch_out = Conv2D(filters=filters, kernel_size=(1, 1), padding='same')(input_tensor)

        # Add main path outputs with the branch outputs
        add = Add()([relu, branch_out])

        return add

    # First level
    level1_out = residual_block(inputs, filters=64)

    # Second level of the residual connection structure
    def level2_block(input_tensor, filters):
        # First block in the second level
        first_block_out = residual_block(input_tensor, filters=filters)
        # Second block in the second level
        second_block_out = residual_block(input_tensor, filters=filters)

        # Add the outputs from the main path and branch of the first block with the main path output of the second block
        add = Add()([first_block_out, second_block_out])

        return add

    # Third level of the residual connection structure
    def level3_block(input_tensor, filters):
        # Global branch
        global_branch = Conv2D(filters=filters, kernel_size=(1, 1), padding='same')(input_tensor)
        # Add the output from the global branch to the result of the second-level residual structure
        add = Add()([global_branch, level2_block(input_tensor, filters=filters)])

        return add

    # Apply second level and third level
    level2_out = level2_block(level1_out, filters=64)
    level3_out = level3_block(level2_out, filters=64)

    # Global average pooling
    avg_pool = MaxPooling2D(pool_size=(4, 4))(level3_out)

    # Fully connected layer
    fc = Dense(units=10, activation='softmax')(avg_pool)

    # Model
    model = keras.Model(inputs=inputs, outputs=fc)

    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model architecture
model.to_json()