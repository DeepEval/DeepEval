import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First branch
    def first_branch(input_tensor):
        conv_dw = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_dw)
        dropout_1 = Dropout(rate=0.5)(conv_1x1)
        batch_norm_1 = BatchNormalization()(dropout_1)
        pool_1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(batch_norm_1)
        return pool_1
    
    first_branch_output = first_branch(input_tensor=input_layer)
    
    # Second branch
    def second_branch(input_tensor):
        conv_dw = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_dw)
        dropout_2 = Dropout(rate=0.5)(conv_1x1)
        batch_norm_2 = BatchNormalization()(dropout_2)
        pool_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(batch_norm_2)
        return pool_2
    
    second_branch_output = second_branch(input_tensor=input_layer)
    
    # Third branch
    def third_branch(input_tensor):
        conv_dw = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv_1x1 = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_dw)
        dropout_3 = Dropout(rate=0.5)(conv_1x1)
        batch_norm_3 = BatchNormalization()(dropout_3)
        pool_3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(batch_norm_3)
        return pool_3
    
    third_branch_output = third_branch(input_tensor=input_layer)
    
    # Concatenate the outputs from all branches
    concatenated_output = Concatenate()(
        [first_branch_output, second_branch_output, third_branch_output])
    
    # Batch normalization and flattening
    batch_norm_concat = BatchNormalization()(concatenated_output)
    flatten_layer = Flatten()(batch_norm_concat)
    
    # Two fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])