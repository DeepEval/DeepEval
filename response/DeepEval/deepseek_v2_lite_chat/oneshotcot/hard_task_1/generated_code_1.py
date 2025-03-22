import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, BatchNormalization, Activation, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Feature extraction paths
    # Path1: Global average pooling followed by two fully connected layers
    avg_pooling = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=1024, activation='relu')(avg_pooling)
    dense2 = Dense(units=512, activation='relu')(dense1)
    
    # Path2: Global max pooling followed by two fully connected layers
    max_pooling = GlobalMaxPooling2D()(input_layer)
    dense3 = Dense(units=1024, activation='relu')(max_pooling)
    dense4 = Dense(units=512, activation='relu')(dense3)
    
    # Concatenate the outputs from both paths
    concat = Concatenate()([dense1, dense2, dense3, dense4])
    
    # Activation for channel attention weights
    attention = Activation('sigmoid')(concat)
    
    # Apply attention weights to the original features
    scaled_features = attention * avg_pooling
    
    # Block 2: Spatial feature extraction
    # Separate average and max pooling
    avg_pooling_block2 = GlobalAveragePooling2D()(input_layer)
    max_pooling_block2 = GlobalMaxPooling2D()(input_layer)
    
    # Concatenate along the channel dimension
    concatenated_features = Concatenate()([avg_pooling_block2, max_pooling_block2])
    
    # 1x1 convolution and sigmoid activation for normalization
    conv1x1 = Conv2D(filters=input_layer.shape[3], kernel_size=(1, 1), padding='same')(input_layer)
    conv_sigmoid = Activation('sigmoid')(conv1x1)
    
    # Multiply normalized features with channel attention features
    combined_features = concatenated_features * conv_sigmoid
    
    # Additional branch with a 1x1 convolutional layer
    branch_output = Conv2D(filters=input_layer.shape[3], kernel_size=(1, 1), padding='same')(input_layer)
    branch_output = Activation('relu')(branch_output)
    final_output = branch_output + combined_features
    
    # Final classification with a fully connected layer
    output_layer = Dense(units=10, activation='softmax')(final_output)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])