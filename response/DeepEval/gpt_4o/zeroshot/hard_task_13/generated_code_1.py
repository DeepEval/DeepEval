from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Multiply, Reshape, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (32, 32, 3)
    num_classes = 10

    # Input layer
    inputs = Input(shape=input_shape)
    
    # Block 1: Multi-branch feature extraction
    conv1x1 = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    conv3x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv5x5 = Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
    maxpool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    
    # Concatenate the outputs of the parallel branches
    concat = Concatenate()([conv1x1, conv3x3, conv5x5, maxpool])
    
    # Block 2: Global Average Pooling and Attention Mechanism
    gap = GlobalAveragePooling2D()(concat)
    
    # Fully connected layers to generate attention weights
    fc1 = Dense(128, activation='relu')(gap)
    fc2 = Dense(concat.shape[-1], activation='sigmoid')(fc1)
    
    # Reshape and multiply to get attention-weighted feature maps
    attention_weights = Reshape((1, 1, concat.shape[-1]))(fc2)
    weighted_features = Multiply()([concat, attention_weights])
    
    # Flatten and output layer
    flatten = Flatten()(weighted_features)
    outputs = Dense(num_classes, activation='softmax')(flatten)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model (uncomment to train)
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))