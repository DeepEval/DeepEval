import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Multiply, Concatenate, AveragePooling2D, Flatten
from tensorflow.keras.models import Model

def dl_model():
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    x = Conv2D(32, (3, 3), padding='same', activation=None)(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Save the initial feature map to multiply with weighted features later
    initial_features = x
    
    # Global average pooling
    x = GlobalAveragePooling2D()(x)
    
    # Fully connected layers to adjust dimensions
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='sigmoid')(x)  # Use sigmoid to get weights between 0 and 1
    
    # Reshape to match the initial feature map's shape
    x = tf.reshape(x, (-1, 1, 1, 32))
    
    # Multiply to create weighted feature maps
    weighted_features = Multiply()([initial_features, x])
    
    # Concatenate weighted features with the input layer
    x = Concatenate()([input_layer, weighted_features])
    
    # Dimensionality reduction and downsampling using 1x1 convolution and average pooling
    x = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    
    # Flatten and final fully connected layer for classification
    x = Flatten()(x)
    output_layer = Dense(10, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()