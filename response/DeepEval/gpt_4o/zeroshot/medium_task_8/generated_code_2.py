import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define the input shape (32, 32, 3) for CIFAR-10 images
    input_shape = (32, 32, 3)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Split the input into three groups along the last dimension
    split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # First group remains unchanged
    first_group = split_groups[0]
    
    # Second group: feature extraction via a 3x3 convolutional layer
    second_group = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_groups[1])
    
    # Combine the second group output with the third group
    combined_second_third = Concatenate(axis=-1)([second_group, split_groups[2]])
    
    # Additional 3x3 convolution on the combined second and third groups
    combined_output = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(combined_second_third)
    
    # Concatenate the outputs of all three groups to form the main path output
    main_path_output = Concatenate(axis=-1)([first_group, second_group, combined_output])
    
    # Branch path: 1x1 convolutional layer to process the input
    branch_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(inputs)
    
    # Fuse the outputs from the main path and the branch path via addition
    fused_output = Add()([main_path_output, branch_output])
    
    # Flatten the combined output
    flattened_output = Flatten()(fused_output)
    
    # Fully connected layer for classification
    final_output = Dense(units=10, activation='softmax')(flattened_output)
    
    # Construct the model
    model = Model(inputs=inputs, outputs=final_output)
    
    return model

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Get the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# You can train the model as follows (uncomment to train):
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.1)

# Evaluate the model (uncomment to evaluate):
scores = model.evaluate(x_test, y_test, verbose=0)
print(f'Test accuracy: {scores[1]}')