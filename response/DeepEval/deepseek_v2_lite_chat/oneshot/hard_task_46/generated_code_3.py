import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups
    channel_indices = Lambda(lambda x: keras.backend.ctc_batch_cost_fn(x, num_chars=3), name='channel_splits')(input_layer)
    
    # First block
    def block1(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv_1x1')(input_tensor[:, :, 0, :])
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_3x3')(input_tensor[:, :, 1, :])
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_5x5')(input_tensor[:, :, 2, :])
        path_concat = Concatenate(name='path_concat_1')(keras.layers.concatenate([path1, path2, path3]))
        
        return path_concat
    
    # Second block
    def block2(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_tensor)
        path_concat = Concatenate(name='path_concat_2')(keras.layers.concatenate([path1, path2, path3]))
        
        return path_concat
    
    # Output layers
    block1_output = block1(channel_indices)
    block2_output = block2(block1_output)
    avg_pool = GlobalAveragePooling2D()(block2_output)
    dense1 = Dense(units=128, activation='relu')(avg_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))