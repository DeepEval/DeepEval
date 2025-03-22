from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, concatenate, Dense
from keras.layers import BatchNormalization, Activation

def dl_model():
    # Parameters
    img_rows, img_cols, img_channels = 28, 28, 1  # MNIST images are 28x28 and grayscale
    num_classes = 10  # 10 classes for MNIST

    # Input layer
    inputs = Input(shape=(img_rows, img_cols, img_channels))
    
    # Block 1
    x = inputs
    for stride in [1, 2, 4]:
        x = Conv2D(32, (3, 3), strides=stride, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.2)(x)
    
    # Flatten and regularize
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    
    # Block 2
    x = Input(shape=(16,))  # 16 features after block 1
    for _ in range(4):
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
    
    # Concatenate features from different paths
    x = concatenate([x[i] for i in range(num_classes)])
    
    # Classification head
    x = Dense(1024, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Model
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

# Instantiate the model
model = dl_model()
model.summary()