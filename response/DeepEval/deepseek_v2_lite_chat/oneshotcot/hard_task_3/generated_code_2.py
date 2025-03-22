import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dropout, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Split input into three groups along the channel dimension
    channel_splits = Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    
    # Process each group through a sequence of <1x1 convolution, 3x3 convolution>
    def process_group(group):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(group)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
        drop1 = Dropout(rate=0.5)(conv2)
        return drop1
    
    # Apply feature extraction to each group
    group_features = [process_group(group) for group in channel_splits]
    
    # Concatenate the features from all groups
    concatenated_features = Concatenate()(group_features)
    
    # Branch pathway: process input through a 1x1 convolution
    branch_conv1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(inputs)
    branch_drop1 = Dropout(rate=0.5)(branch_conv1)
    
    # Combine main and branch pathways
    combined_features = Concatenate()([concatenated_features, branch_drop1])
    
    # Batch normalization and flattening
    bn = BatchNormalization()(combined_features)
    flat = Flatten()(bn)
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=inputs, outputs=output)
    
    return model