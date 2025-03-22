import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, BatchNormalization, ZeroPadding2D
from keras.models import Model
import keras.backend as K

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Average pooling of different scales
    def block1():
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
        conv1_2 = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
        conv1_3 = Conv2D(filters=32, kernel_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
        avg_pool = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
        flatten = Flatten()(avg_pool)
        dropout = Dropout(rate=0.5)(flatten)  # Regularization with dropout
        return K.concatenate([conv1_1, conv1_2, conv1_3, dropout])
    
    # Block 2: Multiple branch connections
    def block2():
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
        path4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
        
        conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        conv2_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        conv2_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
        concat = Concatenate()([path1, path2, path3, conv2_1, conv2_2, conv2_3])
        
        flatten = Flatten()(concat)
        dense1 = Dense(units=128, activation='relu')(flatten)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)
        
        return output_layer
    
    # Model construction
    model = Model(inputs=input_layer, outputs=[block1(), block2()])
    
    return model

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])