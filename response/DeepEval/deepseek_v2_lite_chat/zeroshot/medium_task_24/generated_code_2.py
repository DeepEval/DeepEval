import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dense, Dropout, Flatten

# Number of classes in CIFAR-10
NUM_CLASSES = 10

# Number of filters in the first branch
FILTERS_1 = 32
SIZE_1 = (3, 3)

# Number of filters in the second branch
FILTERS_2 = 64
SIZE_2 = (3, 3)

# Number of filters in the third branch
FILTERS_3 = 64
SIZE_3 = (3, 3)

# Dropout rate
DROPOUT_RATE = 0.2

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # First branch
    x = Conv2D(filters=FILTERS_1, kernel_size=SIZE_1, activation='relu')(inputs)
    x = Conv2D(filters=FILTERS_1, kernel_size=SIZE_1, activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    # Second branch
    x = Conv2D(filters=FILTERS_2, kernel_size=SIZE_2)(inputs)
    x = Conv2D(filters=FILTERS_2, kernel_size=SIZE_2)(x)
    x = Flatten()(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    # Third branch
    x = MaxPooling2D()(inputs)
    x = Dropout(DROPOUT_RATE)(x)
    
    # Concatenate features from all branches
    x = Concatenate()([x, x, x])
    
    # Fully connected layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

return model