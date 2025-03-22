import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path for feature extraction
    def extract_features(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
        pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
        return pool
    
    # Branch path for generating channel weights
    def generate_weights(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=512, activation='relu')(avg_pool)
        dense2 = Dense(units=256, activation='relu')(dense1)
        channel_weights = Dense(32)(dense2)  # Assuming 32 channels for weights
        channel_weights = Dense(32)(channel_weights)  # Reshape channel weights
        return channel_weights
    
    # Construct the model
    main_features = extract_features(input_layer)
    branch_weights = generate_weights(input_layer)
    weighted_features = Concatenate()([main_features, branch_weights])
    
    # Additional fully connected layers for classification
    dense1 = Dense(units=256, activation='relu')(weighted_features)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

# Build the model
model = dl_model()
model.summary()