import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def dl_model():
    # Input shape should be (28, 28, 1) for MNIST images
    input_layer = tf.keras.Input(shape=(28, 28, 1))
    
    # Increase dimensionality with a 1x1 convolutional layer
    x = Conv2D(32, kernel_size=1, activation='relu')(input_layer)
    
    # Depthwise separable convolutional layer
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    
    # Reduce dimensionality with another 1x1 convolutional layer
    x = Conv2D(16, kernel_size=1, activation='relu')(x)
    
    # Apply stride of 2 to reduce spatial size
    x = Conv2D(8, (2, 2), padding='same', activation='relu')(x)
    
    # Max pooling layer
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Flatten the output for the dense layer
    x = Flatten()(x)
    
    # Fully connected layer for classification
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# To test the model
model = dl_model()
model.summary()