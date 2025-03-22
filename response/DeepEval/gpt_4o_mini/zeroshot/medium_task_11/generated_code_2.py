import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model(input_shape=(32, 32, 3), num_classes=10):
    # Input Layer
    inputs = layers.Input(shape=input_shape)
    
    # Initial Convolutional Layer
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Parallel Paths
    # Global Average Pooling Path
    avg_pool_path = layers.GlobalAveragePooling2D()(x)
    avg_pool_path = layers.Dense(128, activation='relu')(avg_pool_path)
    avg_pool_path = layers.Dense(64, activation='relu')(avg_pool_path)

    # Global Max Pooling Path
    max_pool_path = layers.GlobalMaxPooling2D()(x)
    max_pool_path = layers.Dense(128, activation='relu')(max_pool_path)
    max_pool_path = layers.Dense(64, activation='relu')(max_pool_path)

    # Attention Mechanism
    attention_weights = layers.Add()([avg_pool_path, max_pool_path])
    attention_weights = layers.Activation('sigmoid')(attention_weights)
    
    # Apply attention to the original features
    channel_features = layers.Multiply()([x, attention_weights])

    # Spatial Features
    avg_features = layers.GlobalAveragePooling2D()(channel_features)
    max_features = layers.GlobalMaxPooling2D()(channel_features)

    # Concatenate Spatial Features
    spatial_features = layers.Concatenate()([avg_features, max_features])

    # Combine Channel Features with Spatial Features
    combined_features = layers.Multiply()([avg_pool_path, spatial_features])

    # Flatten and Fully Connected Layer
    x = layers.Flatten()(combined_features)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

# Example of how to compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Example of loading and preprocessing the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Model Summary
model.summary()