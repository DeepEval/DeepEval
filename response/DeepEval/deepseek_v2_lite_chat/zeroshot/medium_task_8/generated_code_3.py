import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Add, Concatenate


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the input shape (32x32 images)
input_shape = (32, 32, 3)

# Image data augmentation
datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)
datagen.fit(x_train)


# Main path
def main_path(input_tensor):
    x = Conv2D(32, (3, 3), activation='relu')(input_tensor)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    
    return x

# Branch path
def branch_path(input_tensor):
    x = Conv2D(64, (1, 1), activation='relu')(input_tensor)
    return x

# Fusion layer
def fusion_layer(main_output, branch_output):
    return Add()([main_output, branch_output])

# Final classification layer
def classification_head(input_tensor):
    input_tensor = Flatten()(input_tensor)
    output_tensor = Dense(10, activation='softmax')(input_tensor)
    return output_tensor

# Function to build the model
def dl_model():
    # Input
    inputs = Input(shape=input_shape)
    
    # Main path
    main_output = main_path(inputs)
    
    # Branch path
    branch_output = branch_path(inputs)
    
    # Fusion
    fused_output = fusion_layer(main_output, branch_output)
    
    # Classification head
    output = classification_head(fused_output)
    
    # Model
    model = Model(inputs=[inputs], outputs=[output])
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])