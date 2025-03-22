from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Branch path
    y = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    y = MaxPooling2D(pool_size=(2, 2))(y)
    
    # Combine paths
    combined = Add()([x, y])
    
    # Flatten and fully connected layers
    z = Flatten()(combined)
    z = Dense(128, activation='relu')(z)
    output_layer = Dense(10, activation='softmax')(z)
    
    # Construct model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create and train the model
model = dl_model()
model.summary()

# You can train the model with something like:
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)