from tensorflow.keras.layers import Input, Conv2D, Dropout, Add, Lambda, SeparableConv2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 color channels
    num_classes = 10  # CIFAR-10 has 10 classes
    
    inputs = Input(shape=input_shape)
    
    # First Block
    # Main path
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    dropout1 = Dropout(0.3)(conv1)
    conv2 = Conv2D(3, (3, 3), activation='relu', padding='same')(dropout1)
    
    # Branch path
    branch = inputs
    
    # Add main path and branch path
    added = Add()([conv2, branch])
    
    # Second Block
    def split_and_process(x):
        splits = tf.split(x, num_or_size_splits=3, axis=-1)
        processed = []
        for i, kernel_size in enumerate([(1, 1), (3, 3), (5, 5)]):
            conv = SeparableConv2D(64, kernel_size, activation='relu', padding='same')(splits[i])
            dropout = Dropout(0.3)(conv)
            processed.append(dropout)
        return processed
    
    # Lambda layer to handle custom TensorFlow operations
    processed_splits = Lambda(split_and_process)(added)
    
    # Concatenate processed splits
    concatenated = Concatenate()(processed_splits)
    
    # Output block
    flatten = Flatten()(concatenated)
    outputs = Dense(num_classes, activation='softmax')(flatten)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()