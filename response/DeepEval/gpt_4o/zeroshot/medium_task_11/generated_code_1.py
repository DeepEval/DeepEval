from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply, AveragePooling2D, MaxPooling2D, Concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv_layer = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)

    # Global Average Pooling path
    global_avg_pool = GlobalAveragePooling2D()(conv_layer)
    dense_avg_1 = Dense(64, activation='relu')(global_avg_pool)
    dense_avg_2 = Dense(32, activation='sigmoid')(dense_avg_1)

    # Global Max Pooling path
    global_max_pool = GlobalMaxPooling2D()(conv_layer)
    dense_max_1 = Dense(64, activation='relu')(global_max_pool)
    dense_max_2 = Dense(32, activation='sigmoid')(dense_max_1)

    # Add the outputs of the two paths
    channel_attention = Add()([dense_avg_2, dense_max_2])

    # Apply channel attention weights
    channel_attention_reshaped = Reshape((1, 1, 32))(channel_attention)
    channel_scaled_features = Multiply()([conv_layer, channel_attention_reshaped])

    # Separate average and max pooling
    avg_pool_spatial = AveragePooling2D(pool_size=(2, 2))(channel_scaled_features)
    max_pool_spatial = MaxPooling2D(pool_size=(2, 2))(channel_scaled_features)

    # Concatenate along the channel dimension
    fused_features = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])

    # Combine spatial features with channel features through element-wise multiplication
    combined_features = Multiply()([fused_features, channel_scaled_features])

    # Flatten and final dense layer for classification
    flattened = Flatten()(combined_features)
    output_layer = Dense(10, activation='softmax')(flattened)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()