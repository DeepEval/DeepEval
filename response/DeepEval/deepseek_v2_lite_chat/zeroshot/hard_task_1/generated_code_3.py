import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Concatenate, Activation, Layer
from keras.layers.merge import Add

def dl_model():
    # Input layer
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32
    input_layer = Input(shape=input_shape)
    
    # Block 1: Feature Extraction
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    avg1 = GlobalAveragePooling2D()(pool1)
    fc1 = Dense(128, activation='relu')(avg1)
    
    max1 = GlobalMaxPooling2D()(pool1)
    fc2 = Dense(128, activation='relu')(max1)
    
    # Concatenate the outputs of both paths
    concat = concatenate([fc1, fc2])
    
    # Block 2: Channel Attention
    channel_attention = Dense(128, activation='sigmoid')(concat)
    channel_attention = Activation('sigmoid')(channel_attention)
    
    # Multiply the channel attention weights with the outputs of Block 1
    attention1 = multiply([conv1, channel_attention])
    
    # Average and Max pooling for spatial features
    avg_pool2 = GlobalAveragePooling2D()(pool1)
    max_pool2 = GlobalMaxPooling2D()(pool1)
    
    # Concatenate the outputs of both spatial pooling layers
    concat_spat = concatenate([avg_pool2, max_pool2])
    
    # Spatial attention layer
    spatial_attention = Dense(128, activation='sigmoid')(concat_spat)
    spatial_attention = Activation('sigmoid')(spatial_attention)
    
    # Multiply the spatial attention weights with the outputs of Block 1
    attention2 = multiply([conv1, spatial_attention])
    
    # Add the outputs of Block 1 and Block 2
    output = Add()([attention1, attention2])
    
    # Final classification through a fully connected layer
    output = Dense(10, activation='softmax')(output)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Instantiate the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()