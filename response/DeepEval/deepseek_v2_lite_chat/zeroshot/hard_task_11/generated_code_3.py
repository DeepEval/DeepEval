import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Add, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Input shape
input_shape = (32, 32, 3)

# Create the feature extraction pathway
def extract_features(model_input):
    # 1x1 convolution branch
    conv1x1 = Conv2D(64, (1, 1), activation='relu')(model_input)
    # 1x3 and 3x1 convolution branch
    conv1x3 = Conv2D(64, (1, 3), padding='same', activation='relu')(model_input)
    conv3x1 = Conv2D(64, (3, 1), padding='same', activation='relu')(model_input)
    # Add the two branches
    concat = Concatenate(axis=-1)([conv1x1, conv1x3, conv3x1])
    # Additional 1x1 convolution to match dimensions with input
    conv1x1_output = Conv2D(3, (1, 1), padding='same', activation='sigmoid')(concat)
    return conv1x1_output

# Create the model
def dl_model():
    # Model input
    inputs = Input(shape=input_shape)
    
    # Extract features
    feature_map = extract_features(inputs)
    
    # Fusion with the main pathway
    fused_input = Add()([inputs, feature_map])
    
    # Additional layers for fusion
    fused_layers = Conv2D(64, (1, 1), activation='relu')(fused_input)
    
    # Flatten and fully connected layers for classification
    x = Flatten()(fused_layers)
    output = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=output)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build and return the model
model = dl_model()
model.summary()