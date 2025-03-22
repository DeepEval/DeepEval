from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Concatenate, Activation, MaxPooling2D, Flatten
from keras.layers import BatchNormalization, LeakyReLU

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Number of classes
num_classes = 10

# Function to create the model
def dl_model():
    # Input layer
    input_layer = Input(shape=x_train[0].shape)
    
    # First path: Convolutional layer with global average pooling
    conv_avg = Conv2D(32, (3, 3), padding='same')(input_layer)
    conv_avg = BatchNormalization()(conv_avg)
    conv_avg = LeakyReLU()(conv_avg)
    avg_pool = GlobalAveragePooling2D()(conv_avg)
    
    # Second path: Convolutional layer with global max pooling
    conv_max = Conv2D(32, (3, 3), padding='same')(input_layer)
    conv_max = BatchNormalization()(conv_max)
    conv_max = LeakyReLU()(conv_max)
    max_pool = GlobalMaxPooling2D()(conv_max)
    
    # Concatenate the outputs from both paths
    concat = Concatenate()([avg_pool, max_pool])
    
    # Fully connected layers
    dense1 = Dense(512, activation='relu')(concat)
    dense2 = Dense(256, activation='relu')(dense1)
    
    # Attention mechanism
    attention_weight = Dense(1, activation='sigmoid')(dense2)
    attention_weight = Activation('sigmoid')(attention_weight)
    
    # Processed features with attention weights
    processed_features = Add()([conv_avg, conv_max * attention_weight])
    
    # Average and max pooling for spatial features
    avg_pool_spatial = BatchNormalization()(GlobalAveragePooling2D()(processed_features))
    max_pool_spatial = BatchNormalization()(GlobalMaxPooling2D()(processed_features))
    
    # Concatenate spatial and channel features
    fused_features = Concatenate()([avg_pool_spatial, max_pool_spatial])
    
    # Fully connected layer
    output = Dense(num_classes, activation='softmax')(fused_features)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Build the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()