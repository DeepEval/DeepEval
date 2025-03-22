import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Multiply, Flatten, Concatenate

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have shape 32x32 with 3 channels (RGB)
    
    # Initial convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Parallel paths for channel attention
    avg_pooling = GlobalAveragePooling2D()(conv)
    dense1_avg = Dense(units=64, activation='relu')(avg_pooling)
    dense2_avg = Dense(units=32, activation='sigmoid')(dense1_avg)  # channel attention weights from avg pooling

    max_pooling = GlobalMaxPooling2D()(conv)
    dense1_max = Dense(units=64, activation='relu')(max_pooling)
    dense2_max = Dense(units=32, activation='sigmoid')(dense1_max)  # channel attention weights from max pooling

    # Combine channel attention weights
    attention_weights = Add()([dense2_avg, dense2_max])
    attention_weights = tf.expand_dims(attention_weights, axis=1)  # Expand dimensions to match original feature shape
    attention_weights = tf.expand_dims(attention_weights, axis=1)  # Expand again for multiplication

    # Apply channel attention weights
    channel_attention = Multiply()([conv, attention_weights])

    # Extract spatial features using average and max pooling
    spatial_avg = GlobalAveragePooling2D()(channel_attention)
    spatial_max = GlobalMaxPooling2D()(channel_attention)

    # Concatenate spatial features
    fused_features = Concatenate()([spatial_avg, spatial_max])

    # Final processing
    flatten = Flatten()(fused_features)
    output_layer = Dense(units=10, activation='softmax')(flatten)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model