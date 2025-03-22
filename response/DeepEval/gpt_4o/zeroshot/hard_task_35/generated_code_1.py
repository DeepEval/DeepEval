import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Concatenate, Flatten
from tensorflow.keras.models import Model

def attention_block(input_tensor):
    # Global average pooling
    x = GlobalAveragePooling2D()(input_tensor)
    
    # First fully connected layer
    x = Dense(units=x.shape[-1] // 2, activation='relu')(x)  # Reduce the dimensionality
    
    # Second fully connected layer
    x = Dense(units=input_tensor.shape[-1], activation='sigmoid')(x)
    
    # Reshape to match input dimensions for channel-wise multiplication
    x = tf.reshape(x, [-1, 1, 1, input_tensor.shape[-1]])
    
    # Element-wise multiplication
    x = Multiply()([input_tensor, x])
    
    return x

def dl_model():
    # Define input
    input_shape = (32, 32, 3)
    inputs = Input(shape=input_shape)
    
    # Branch 1
    branch1 = attention_block(inputs)
    
    # Branch 2
    branch2 = attention_block(inputs)
    
    # Concatenate the outputs of the two branches
    concatenated = Concatenate()([branch1, branch2])
    
    # Flatten the concatenated outputs
    flattened = Flatten()(concatenated)
    
    # Fully connected layer for classification
    outputs = Dense(units=10, activation='softmax')(flattened)  # CIFAR-10 has 10 classes
    
    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
# model = dl_model()
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.summary()