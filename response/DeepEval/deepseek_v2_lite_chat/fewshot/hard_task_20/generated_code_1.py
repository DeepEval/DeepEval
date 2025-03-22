import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(input_tensor):
        # Split into three groups
        group1, group2, group3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))(input_tensor)
        
        # Convolutional layers for each group
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(group1)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(group2)
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same')(group3)
        
        # Concatenate the outputs
        concat = Concatenate()([conv1, conv2, conv3])
        
        # Flatten and pass through fully connected layers
        flatten = Flatten()(concat)
        dense1 = Dense(units=128, activation='relu')(flatten)
        dense2 = Dense(units=64, activation='relu')(dense1)
        
        # Output layer
        output_layer = Dense(units=10, activation='softmax')(dense2)
        
        # Define the model
        model = Model(inputs=input_layer, outputs=output_layer)
        return model
    
    # Branch path
    def branch_path(input_tensor):
        # 1x1 convolutional layer
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_tensor)
        
        # Flatten and pass through fully connected layers
        flatten = Flatten()(conv1)
        dense = Dense(units=128, activation='relu')(flatten)
        
        # Output layer
        output_layer = Dense(units=10, activation='softmax')(dense)
        
        # Define the model for the branch
        branch_model = Model(inputs=input_tensor, outputs=output_layer)
        return branch_model
    
    # Combine the main path and branch path
    main_model = main_path(input_layer)
    branch_model = branch_path(input_layer)
    combined_model = keras.Model(inputs=input_layer, outputs=main_model(Concatenate()([branch_model.output, main_model.output])))
    
    return combined_model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])