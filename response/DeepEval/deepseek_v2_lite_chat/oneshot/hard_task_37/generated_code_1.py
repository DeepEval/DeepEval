import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Number of classes
    num_classes = 10
    
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Branch 1
    branch1_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1_block1 = block(branch1_conv)
    branch1_block2 = block(branch1_block1)
    branch1_block3 = block(branch1_block2)
    branch1_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch1_block3)
    
    # Branch 2
    branch2_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2_block1 = block(branch2_conv)
    branch2_block2 = block(branch2_block1)
    branch2_block3 = block(branch2_block2)
    branch2_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch2_block3)
    
    # Concatenate outputs from both branches
    concat = Concatenate(axis=-1)([branch1_pooling, branch2_pooling])
    
    # Additional convolutional layer for direct branch
    direct_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    direct_block1 = block(direct_conv)
    direct_block2 = block(direct_block1)
    direct_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(direct_block2)
    
    # Combine all paths using addition
    combined = keras.layers.Add()([concat, direct_pooling])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(combined)
    flatten = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=num_classes, activation='softmax')(dense2)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

def block(input_tensor):
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
    path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(path2)
    pooling = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(path3)
    return pooling

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])