import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Flatten
from tensorflow.keras.applications import ResNet50

def dl_model():
    # Load a pre-trained ResNet50 model as a base model
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
    
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Branch path
    branch = Conv2D(filters=3, kernel_size=(3, 3), strides=1, padding='same')(inputs)
    
    # Main path
    main = base_model(inputs)
    main = GlobalAveragePooling2D()(main)
    main = Dense(128, activation='relu')(main)
    main = Dense(128, activation='relu')(main)
    weights = Dense(3, activation='sigmoid')(main)
    weights = tf.reshape(weights, (-1, 32, 32, 3))
    weighted_features = tf.multiply(main, weights)
    
    # Combine outputs from both paths
    combined = Add()([branch, weighted_features])
    
    # Final classification layers
    x = Flatten()(combined)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the final model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])