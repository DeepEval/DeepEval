import keras
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three channel groups
    channel_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))(input_layer)
    
    # Feature extraction for each channel group
    def extract_features(channel_group):
        conv1 = Conv2D(filters=32, kernel_size=1, activation='relu', padding='same')(channel_group)
        conv2 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=5, activation='relu', padding='same')(conv2)
        pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)
        return pool
    
    # Output from each channel group
    features_1 = extract_features(channel_groups[0])
    features_2 = extract_features(channel_groups[1])
    features_3 = extract_features(channel_groups[2])
    
    # Concatenate features
    concatenated_features = Add()([features_1, features_2, features_3])
    
    # Flatten and pass through fully connected layers
    flat = Flatten()(concatenated_features)
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()