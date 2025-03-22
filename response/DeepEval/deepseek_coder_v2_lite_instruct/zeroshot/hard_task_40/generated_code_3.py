import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Concatenate, Dropout, Reshape

def dl_model():
    # Define the input layer
    inputs = Input(shape=(28, 28, 1))

    # First block
    x = Conv2D(64, (1, 1), activation='relu')(inputs)
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1)(x)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2)(x)
    pool4 = AveragePooling2D(pool_size=(4, 4), strides=4)(x)
    
    # Flatten and concatenate the outputs of the pooling layers
    z = Flatten()(pool1)
    y = Flatten()(pool2)
    w = Flatten()(pool4)
    concat_features = Concatenate()([z, y, w])
    
    # Reshape the concatenated vector into a 4-dimensional tensor
    reshaped = Reshape((1, 1, 9))(concat_features)
    
    # Second block
    # Path 1
    path1 = Conv2D(32, (1, 1), activation='relu')(reshaped)
    path1 = Dropout(0.5)(path1)
    
    # Path 2
    path2 = Conv2D(32, (1, 1), activation='relu')(reshaped)
    path2 = Conv2D(32, (3, 3), activation='relu')(path2)
    path2 = Conv2D(32, (3, 3), activation='relu')(path2)
    path2 = Dropout(0.5)(path2)
    
    # Path 3
    path3 = Conv2D(32, (1, 1), activation='relu')(reshaped)
    path3 = Conv2D(32, (3, 3), activation='relu')(path3)
    path3 = Dropout(0.5)(path3)
    
    # Path 4
    path4 = AveragePooling2D(pool_size=(1, 1), strides=1)(reshaped)
    path4 = Conv2D(32, (1, 1), activation='relu')(path4)
    path4 = Dropout(0.5)(path4)
    
    # Concatenate outputs from all paths along the channel dimension
    combined = Concatenate(axis=-1)([path1, path2, path3, path4])
    
    # Flatten the concatenated tensor
    combined_flat = Flatten()(combined)
    
    # Output layer
    outputs = Dense(10, activation='softmax')(combined_flat)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()