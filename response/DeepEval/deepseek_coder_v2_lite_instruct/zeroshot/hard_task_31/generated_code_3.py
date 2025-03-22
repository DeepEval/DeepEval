import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, Lambda, SeparableConv2D, Flatten, Dense

def dl_model():
    input_shape = (32, 32, 3)
    
    # First block
    def block1(x):
        # Main path
        main_path = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        main_path = Conv2D(32, (3, 3), activation='relu', padding='same')(main_path)
        main_path = Dropout(0.5)(main_path)
        main_path = Conv2D(32, (3, 3), activation='relu', padding='same')(main_path)
        
        # Branch path
        branch_path = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
        
        # Add outputs from both paths
        output = Add()([main_path, branch_path])
        return output
    
    # Second block
    def block2(x):
        # Split the input into three groups
        split_1 = Lambda(lambda y: tf.split(y, num_or_size_splits=3, axis=-1))(x)
        split_2 = Lambda(lambda y: tf.split(y, num_or_size_splits=3, axis=-1))(x)
        
        # Process each group with separable convolutional layers
        output_1 = SeparableConv2D(32, (1, 1), activation='relu', padding='same')(split_1)
        output_1 = Dropout(0.5)(output_1)
        
        output_2 = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(split_2)
        output_2 = Dropout(0.5)(output_2)
        
        output_3 = SeparableConv2D(32, (5, 5), activation='relu', padding='same')(split_2)
        output_3 = Dropout(0.5)(output_3)
        
        # Concatenate the outputs
        output = tf.keras.layers.Concatenate()([output_1, output_2, output_3])
        return output
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Apply the blocks
    x = block1(inputs)
    x = block2(x)
    
    # Flatten and fully connected layer
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Create the model
model = dl_model()
model.summary()