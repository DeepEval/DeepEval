import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add 1x1 convolutional layers
    conv1x1_1 = Conv2D(32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv1x1_2 = Conv2D(32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Step 3: Add 3x1 convolutional layer
    conv3x1 = Conv2D(32, kernel_size=(3, 1), activation='relu')(conv1x1_1)
    
    # Step 4: Add 1x3 convolutional layer
    conv1x3 = Conv2D(32, kernel_size=(1, 3), activation='relu')(conv1x1_2)
    
    # Step 5: Restore the number of channels to match the input's channel count using another 1x1 convolutional layer
    restore_channels = Conv2D(32, kernel_size=(1, 1), activation='relu')(conv3x1)
    
    # Step 6: Add dropout layers to mitigate overfitting
    dropout_conv3x1 = Dropout(0.25)(conv3x1)
    dropout_conv1x3 = Dropout(0.25)(conv1x3)
    dropout_restore_channels = Dropout(0.25)(restore_channels)
    
    # Step 7: Combine the processed features with the original input via addition
    added = Add()([dropout_conv3x1, dropout_conv1x3, dropout_restore_channels, input_layer])
    
    # Step 8: Add flatten layer
    flatten_layer = Flatten()(added)
    
    # Step 9: Add dense layer to produce the final probability distribution
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Build the model
    model = Model(inputs=input_layer, outputs=dense_layer)
    
    return model