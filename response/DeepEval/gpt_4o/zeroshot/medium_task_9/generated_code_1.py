from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def basic_block(x, filters, kernel_size=(3, 3), strides=(1, 1), padding='same'):
    # Main path
    conv = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    norm = BatchNormalization()(conv)
    act = ReLU()(norm)
    
    # Branch path
    branch = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding=padding)(x)

    # Feature fusion
    out = Add()([act, branch])
    return out

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer to reduce the dimensionality
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Two consecutive basic blocks
    x = basic_block(x, 16)
    x = basic_block(x, 16)

    # Final feature fusion with a branch
    # Branch path
    branch = Conv2D(16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    
    # Combine paths
    x = Add()([x, branch])
    
    # Average pooling
    x = AveragePooling2D(pool_size=(2, 2))(x)

    # Flatten and fully connected layer
    x = Flatten()(x)
    output_layer = Dense(10, activation='softmax')(x)

    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Return the constructed model
    return model

# Optional: Compile and print model summary for verification
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()