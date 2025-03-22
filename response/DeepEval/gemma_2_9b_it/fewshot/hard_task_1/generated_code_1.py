import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Concatenate, Activation, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial Convolutional Layer
    conv_layer = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)

    # Block 1: Channel Attention
    path1 = GlobalAveragePooling2D()(conv_layer)
    path1 = Dense(units=64, activation='relu')(path1)
    path1 = Dense(units=3, activation='sigmoid')(path1)

    path2 = GlobalMaxPooling2D()(conv_layer)
    path2 = Dense(units=64, activation='relu')(path2)
    path2 = Dense(units=3, activation='sigmoid')(path2)

    attention_weights = Add()([path1, path2])

    # Apply attention weights
    attended_features = Lambda(lambda x: x * attention_weights)(conv_layer)

    # Block 2: Spatial Feature Extraction
    avg_pool = GlobalAveragePooling2D()(attended_features)
    max_pool = GlobalMaxPooling2D()(attended_features)
    spatial_features = Concatenate(axis=3)([avg_pool, max_pool])

    spatial_features = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(spatial_features)
    spatial_features = Activation('sigmoid')(spatial_features)

    # Element-wise Multiplication
    features = Multiply()([spatial_features, attended_features])

    # Final Branch
    final_branch = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(features)

    # Add and activate
    output = Add()([features, final_branch])
    output = Activation('relu')(output)

    # Classification
    output_layer = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model