import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, GlobalAveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
        return max_pool
    
    main_pool = main_path(input_tensor=input_layer)
    
    # Branch path
    def branch_path(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=512, activation='relu')(avg_pool)
        dense2 = Dense(units=256, activation='relu')(dense1)
        return dense2
    
    branch_output = branch_path(input_tensor=main_pool)
    
    # Channel weights
    weight_tensor = Dense(units=main_pool.shape[1] * branch_output.shape[1], activation='sigmoid')(branch_output)
    weight_tensor = Reshape((main_pool.shape[1], branch_output.shape[1]))(weight_tensor)
    weight_tensor = Activation('sigmoid')(weight_tensor)
    
    # Output addition
    final_output = Concatenate()([main_pool, main_pool * weight_tensor])
    final_output = Dense(units=10, activation='softmax')(final_output)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=final_output)
    
    return model

# Instantiate the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])