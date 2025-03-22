import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Extraction of deep features
    def feature_extraction(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        return conv3, pool3
    
    conv3, pool3 = feature_extraction(inputs)
    
    # Enhancement of generalization capabilities
    def enhance_generalization(input_tensor):
        dropout1 = Dropout(0.5)(input_tensor)
        conv4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(dropout1)
        
        dropout2 = Dropout(0.5)(conv4)
        conv5 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(dropout2)
        
        return conv5
    
    conv5 = enhance_generalization(pool3)
    
    # Upsampling with skip connections
    def upsample_with_skip(input_tensor, conv3):
        transpose = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(input_tensor)
        concat = concatenate([transpose, conv3], axis=3)
        conv6 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(concat)
        
        transpose = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv6)
        concat = concatenate([transpose, conv5], axis=3)
        conv7 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(concat)
        
        return conv7
    
    conv7 = upsample_with_skip(inputs, conv3)
    
    # Final layer for classification
    def classification_layer(input_tensor):
        conv8 = Conv2D(filters=10, kernel_size=(1, 1), padding='same', activation='softmax')(input_tensor)
        return conv8
    
    output = classification_layer(conv7)
    
    # Model
    model = Model(inputs=inputs, outputs=output)
    
    return model