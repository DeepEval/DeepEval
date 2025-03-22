import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main pathway
    def main_path(input_tensor):
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        path1 = Conv2D(filters=64, kernel_size=(1x1), activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1x3), activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(3x1), activation='relu')(input_tensor)
        conv_concat = Concatenate()([conv1x1, path1, path2, path3])
        pool = MaxPooling2D(pool_size=(2, 2))(conv_concat)
        batch_norm = BatchNormalization()(pool)
        flatten = Flatten()(batch_norm)
        dense1 = Dense(units=128, activation='relu')(flatten)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)
        
        return output_layer
    
    # Direct connection branch
    def direct_connection(input_tensor):
        direct_conv = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        dense = Dense(units=64, activation='relu')(direct_conv)
        output_layer = Dense(units=10, activation='softmax')(dense)
        
        return output_layer
    
    # Model branches
    main_branch = main_path(input_layer)
    direct_branch = direct_connection(input_layer)
    
    # Additive fusion
    output = Add()([main_branch, direct_branch])
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Instantiate the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])