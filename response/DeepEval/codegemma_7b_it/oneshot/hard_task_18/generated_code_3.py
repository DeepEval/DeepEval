import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
        
        input_layer = Input(shape=(32, 32, 3))
        
        # First block
        conv_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
        conv_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_1)
        avg_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_2)
        
        # Residual connection
        residual = avg_pool
        
        # Second block
        conv_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool)
        conv_4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_3)
        avg_pool_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_4)
        
        # Global average pooling for channel weights
        gap = GlobalAveragePooling2D()(avg_pool_2)
        
        # Fully connected layers to refine channel weights
        dense_1 = Dense(units=64, activation='relu')(gap)
        dense_2 = Dense(units=64, activation='relu')(dense_1)
        
        # Reshape channel weights and multiply by input
        channel_weights = Dense(units=64, activation='sigmoid')(dense_2)
        reshaped_weights = Reshape((1, 1, 64))(channel_weights)
        output = Multiply()([avg_pool_2, reshaped_weights])
        
        # Flattening and classification layer
        flatten = Flatten()(output)
        output_layer = Dense(units=10, activation='softmax')(flatten)
        
        model = keras.Model(inputs=input_layer, outputs=output_layer)
        
        return model