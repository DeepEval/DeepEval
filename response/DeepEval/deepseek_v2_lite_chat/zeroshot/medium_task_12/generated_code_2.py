import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, concatenate, Dense, Flatten
from keras.optimizers import Adam

def dl_model():
    # Number of output classes for CIFAR-10 (10)
    num_classes = 10

    # Input shape based on the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Block 1
    x = Conv2D(32, (3, 3), activation=ReLU(), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation=ReLU(), padding='same')(x)
    x = BatchNormalization()(x)
    
    # Block 2
    x = Conv2D(64, (3, 3), activation=ReLU(), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation=ReLU(), padding='same')(x)
    x = BatchNormalization()(x)
    
    # Block 3
    x = Conv2D(128, (3, 3), activation=ReLU(), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation=ReLU(), padding='same')(x)
    x = BatchNormalization()(x)
    
    # Concatenate the outputs from each block
    combined = concatenate([x, x, x])  # Concatenate along the channel dimension

    # Flatten and pass through two dense layers
    x = Flatten()(combined)
    x = Dense(512, activation=ReLU(), kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = Dense(256, activation=ReLU(), kernel_regularizer=keras.regularizers.l2(0.001))(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Optional: Create the model and print the summary
model = dl_model()
model.summary()