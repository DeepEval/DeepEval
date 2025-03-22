import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(inputs)
    
    # Process each group through a series of convolutions
    processed_groups = []
    for group in split:
        # 1x1 convolution
        x = Conv2D(32, kernel_size=(1, 1), activation='relu')(group)
        # 3x3 convolution
        x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
        # Another 1x1 convolution
        x = Conv2D(32, kernel_size=(1, 1), activation='relu')(x)
        processed_groups.append(x)
    
    # Combine the outputs from the three groups using addition
    combined_main_path = Add()(processed_groups)
    
    # Add the processed groups to the original input
    main_path = Add()([combined_main_path, inputs])
    
    # Flatten the combined features
    flattened = Flatten()(main_path)
    
    # Fully connected layer for classification
    outputs = Dense(10, activation='softmax')(flattened)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()