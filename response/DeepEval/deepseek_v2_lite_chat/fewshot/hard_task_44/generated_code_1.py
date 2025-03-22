import keras
from keras.layers import Input, Lambda, Concatenate, Conv2D, MaxPooling2D, Flatten, Dense, Reshape
from keras.models import Model
import tensorflow as tf

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block1(input_tensor):
        # Split the input into three groups
        groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Process each group with different convolutional layers
        conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(groups[0])
        conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(groups[1])
        conv1_3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(groups[2])
        
        # Dropout to reduce overfitting
        conv1_1 = tf.keras.layers.Dropout(rate=0.2)(conv1_1)
        conv1_2 = tf.keras.layers.Dropout(rate=0.2)(conv1_2)
        conv1_3 = tf.keras.layers.Dropout(rate=0.2)(conv1_3)
        
        # Concatenate the outputs from the three groups
        concat_layer = Concatenate(axis=-1)([conv1_1, conv1_2, conv1_3])
        return concat_layer
    
    # Block 2
    def block2(input_tensor):
        # Four branches for different feature extraction
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(input_tensor)
        branch4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor)
        branch4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch4)
        
        # Concatenate the outputs from all branches
        concat_layer = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])
        
        # Flatten and fully connected layers for classification
        flatten = Flatten()(concat_layer)
        dense = Dense(units=128, activation='relu')(flatten)
        output = Dense(units=10, activation='softmax')(dense)
        
        # Model architecture
        model = Model(inputs=input_tensor, outputs=output)
        return model
    
    # Apply block 1
    block1_output = block1(input_layer)
    
    # Apply block 2
    model = block2(block1_output)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])