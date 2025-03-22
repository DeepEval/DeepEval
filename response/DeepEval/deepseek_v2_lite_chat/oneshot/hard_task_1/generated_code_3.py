import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Activation, Dense, Flatten, MaxPooling2D
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layer with output channels adjusted to match input image channels
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Block 1: Parallel feature extraction paths
    path1 = GlobalAveragePooling2D()(conv1)
    path1 = Dense(units=128, activation='relu')(path1)
    
    path2 = GlobalMaxPooling2D()(conv1)
    path2 = Dense(units=128, activation='relu')(path2)
    
    # Concatenate the outputs from both paths
    concat_layer = Concatenate()([path1, path2])
    
    # Attention mechanism to generate channel attention weights
    attention_layer = Dense(units=1, activation='sigmoid')(concat_layer)
    concat_layer = keras.layers.multiply([concat_layer, attention_layer])
    
    # Block 2: Spatial feature extraction
    avg_pool = MaxPooling2D(pool_size=(2, 2))(conv1)
    max_pool = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    concat_spatial = Concatenate()([avg_pool, max_pool])
    concat_spatial = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(concat_spatial)
    sigmoid = Activation('sigmoid')(concat_spatial)
    
    # Element-wise multiplication of attention weights with spatial features
    concat_spatial = keras.layers.multiply([concat_spatial, sigmoid])
    
    # Additional branch to ensure output channels match input channels
    additional_branch = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    additional_branch = Flatten()(additional_branch)
    additional_branch = Dense(units=10, activation='softmax')(additional_branch)
    
    # Combine the main path with the additional branch
    model = keras.layers.add([concat_layer, concat_spatial, additional_branch])
    model = Dense(units=10, activation='softmax')(model)
    
    # Return the constructed model
    return Model(inputs=input_layer, outputs=model)

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])