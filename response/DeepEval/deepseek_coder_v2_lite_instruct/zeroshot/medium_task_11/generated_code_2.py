import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Add, Activation, Multiply
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    
    # First path with GlobalAveragePooling2D followed by two fully connected layers
    path_1 = GlobalAveragePooling2D()(x)
    path_1 = Dense(128, activation='relu')(path_1)
    path_1 = Dense(64, activation='relu')(path_1)
    
    # Second path with GlobalMaxPooling2D followed by two fully connected layers
    path_2 = GlobalMaxPooling2D()(x)
    path_2 = Dense(128, activation='relu')(path_2)
    path_2 = Dense(64, activation='relu')(path_2)
    
    # Adding the outputs from the two paths
    added = Add()([path_1, path_2])
    weights = Activation('sigmoid')(added)
    
    # Applying the channel attention weights to the original features
    channel_attention = Multiply()([x, weights])
    
    # Separate average and max pooling operations to extract spatial features
    spatial_1 = AveragePooling2D((4, 4))(channel_attention)
    spatial_2 = MaxPooling2D((4, 4))(channel_attention)
    
    # Concatenating the spatial features along the channel dimension
    spatial_features = tf.concat([spatial_1, spatial_2], axis=-1)
    
    # Flattening the spatial features
    spatial_features = Flatten()(spatial_features)
    
    # Combining the spatial features with the channel features through element-wise multiplication
    combined_features = Multiply()([channel_attention, spatial_features])
    
    # Final fully connected layer
    output_layer = Dense(10, activation='softmax')(combined_features)
    
    # Constructing the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()