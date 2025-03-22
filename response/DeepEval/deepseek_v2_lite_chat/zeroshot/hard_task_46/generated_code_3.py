import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, Lambda, concatenate, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1)
datagen.fit(x_train)

# Function to define the model
def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split channel into three groups
    group1, group2, group3 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Convolutional layers for each group
    conv1 = SeparableConv2D(32, (1, 1), padding='same')(group1)
    conv2 = SeparableConv2D(32, (3, 3), padding='same')(group2)
    conv3 = SeparableConv2D(32, (5, 5), padding='same')(group3)
    
    # Concatenate the outputs from the three groups
    x = concatenate([conv1, conv2, conv3])
    
    # Second block for enhanced feature extraction
    x = SeparableConv2D(64, (3, 3), padding='same')(x)
    x = SeparableConv2D(64, (3, 3), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = SeparableConv2D(64, (3, 3), padding='same')(x)
    x = SeparableConv2D(64, (3, 3), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = SeparableConv2D(64, (3, 3), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Global average pooling
    x = GlobalAveragePooling2D()(x)
    
    # Dense layer for classification
    output = Dense(10, activation='softmax')(x)
    
    # Model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)