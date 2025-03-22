from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the input shape
input_shape = (32, 32, 3)

# Functional model API
def dl_model():
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First level residual block
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    branch_output = Add()([x, inputs])
    
    # Second level residual block
    x = Conv2D(64, (3, 3), activation='relu')(branch_output)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    branch_output = Add()([x, branch_output])
    
    # Third level residual block
    x = Conv2D(128, (3, 3), activation='relu')(branch_output)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    global_output = GlobalAveragePooling2D()(x)
    output = Dense(10, activation='softmax')(global_output)
    
    # Create the model
    model = Model(inputs=inputs, outputs=output)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build the model
model = dl_model()

# Display model summary
model.summary()