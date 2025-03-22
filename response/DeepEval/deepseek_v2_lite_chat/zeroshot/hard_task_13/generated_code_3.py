from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the functional model
def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First block of branches
    branch1 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(64, (3, 3), activation='relu')(input_layer)
    branch3 = Conv2D(64, (5, 5), activation='relu')(input_layer)
    branch4 = MaxPooling2D(pool_size=(3, 3))(input_layer)
    
    # Concatenate features from different branches
    concatenated = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])
    
    # Second block for dimensionality reduction
    pooled_features = GlobalAveragePooling2D()(concatenated)
    
    # Fully connected layers
    fc1 = Dense(512, activation='relu')(pooled_features)
    output = Dense(10, activation='softmax')(fc1)  # Assuming 10 classes
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model (simplified example)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)