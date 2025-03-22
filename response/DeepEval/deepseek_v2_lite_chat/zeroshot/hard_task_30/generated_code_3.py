import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, Add, concatenate, Lambda, Dense, Flatten
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Function to create the dual-path model
def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Dual-path model
    def dual_path(input_layer):
        # Main path
        x = Conv2D(32, (3, 3), padding='same')(input_layer)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        
        # Branch path
        branch_x = input_layer

        # Combine paths
        combined = Add()([x, branch_x])
        combined = Conv2D(64, (3, 3), padding='same')(combined)
        combined = LeakyReLU(alpha=0.01)(combined)
        combined = BatchNormalization()(combined)
        
        return combined
    
    main_path = dual_path(input_layer)
    
    # Split input into three groups
    split1 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(main_path)
    split2 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(main_path)
    split3 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(main_path)
    
    # Different kernel sizes for depthwise separable convolutions
    def depthwise_separable_layers(input_layer):
        x = split1(input_layer)
        x = Conv2D(16, (1, 1), use_depthwise=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        return x
    
    def large_kernel_layer(input_layer):
        x = split2(input_layer)
        x = Conv2D(16, (3, 3), use_depthwise=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (5, 5), padding='same')(x)
        x = Activation('relu')(x)
        return x
    
    def large_kernel_layer2(input_layer):
        x = split3(input_layer)
        x = Conv2D(16, (5, 5), use_depthwise=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        return x
    
    three_groups = [depthwise_separable_layers(x) for x in main_path]
    large_kernel_groups = [large_kernel_layer(x) for x in main_path]
    large_kernel_groups2 = [large_kernel_layer2(x) for x in main_path]
    
    # Concatenate features from different kernel sizes
    concatenated_features = concatenate([x for x in three_groups + large_kernel_groups + large_kernel_groups2])
    
    # Output layer
    output_layer = Dense(10, activation='softmax')(concatenated_features)
    
    # Model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build and train the model
model = dl_model()
model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)