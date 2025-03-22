from keras.models import Model
from keras.layers import Input, Conv2D, Add, concatenate, Dense, Flatten

def dl_model():
    # Define the main path of the model
    input_main = Input(shape=(32, 32, 64))
    
    # 1x1 convolutional layer for dimensionality reduction
    conv1x1_main = Conv2D(filters=128, kernel_size=1, padding='same')(input_main)
    
    # Two parallel convolutional layers with 1x1 and 3x3 kernels
    conv1x1_branch = Conv2D(filters=128, kernel_size=1, padding='same')(input_main)
    conv3x3_branch = Conv2D(filters=128, kernel_size=3, padding='same')(input_main)
    
    # Concatenate the outputs of the two main path layers
    concat = concatenate([conv1x1_main, conv1x1_branch, conv3x3_branch])
    
    # Process the branch path
    input_branch = Conv2D(filters=128, kernel_size=3, padding='same')(input_main)
    
    # Combine the outputs from the main and branch paths
    combined = Add()([concat, input_branch])
    
    # Flatten the combined output
    flat = Flatten()(combined)
    
    # Two fully connected layers for classification
    dense1 = Dense(1024, activation='relu')(flat)
    output = Dense(10, activation='softmax')(dense1)  # Assuming a classification task with 10 classes
    
    # Model construction
    model = Model(inputs=[input_main, input_branch], outputs=[output])
    
    return model

# Example usage:
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Fit the model (assuming you have the actual training data and labels)
model.fit([X_train_flattened, X_train_branch], y_train, validation_data=([X_val_flattened, X_val_branch], y_val))