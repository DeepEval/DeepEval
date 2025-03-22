import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups for different kernel sizes
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))(input_layer)
    
    # Block 1
    def block1(input_tensor):
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_tensor[0])
        conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor[1])
        conv1_3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(input_tensor[2])
        drop1 = Dropout(0.2)(conv1_1)
        drop2 = Dropout(0.2)(concatenate([conv1_2, conv1_3]))
        return drop2
    
    # Block 2
    def block2(input_tensor):
        conv2_1 = Conv2D(filters=128, kernel_size=(1, 1), padding='same')(input_tensor)
        conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv2_3 = Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu')(input_tensor)
        conv2_4 = MaxPooling2D(pool_size=(2, 2))(input_tensor)
        drop2 = Dropout(0.2)(conv2_1)
        drop2 = Dropout(0.2)(concatenate([drop2, conv2_2, conv2_3, conv2_4]))
        return drop2
    
    # Combine outputs from both blocks
    combined = Concatenate()([block1(input_tensor=split_layer), block2(input_tensor=split_layer)])
    
    # Flatten and fully connected layers
    flatten = Flatten()(combined)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])