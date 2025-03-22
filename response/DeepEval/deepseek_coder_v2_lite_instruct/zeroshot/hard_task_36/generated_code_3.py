import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    # Define the input layer
    inputs = Input(shape=(28, 28, 1))
    
    # Main pathway
    main_pathway = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    main_pathway = Conv2D(32, kernel_size=(1, 1), activation='relu')(main_pathway)
    main_pathway = Conv2D(32, kernel_size=(1, 1), activation='relu')(main_pathway)
    main_pathway = MaxPooling2D(pool_size=(2, 2))(main_pathway)
    main_pathway = Dropout(0.5)(main_pathway)
    
    # Branch pathway
    branch_pathway = Conv2D(32, kernel_size=(3, 3), activation='relu')(main_pathway)
    
    # Fuse the pathways
    fused = Add()([main_pathway, branch_pathway])
    
    # Global average pooling
    fused = GlobalAveragePooling2D()(fused)
    
    # Flatten
    fused = Flatten()(fused)
    
    # Fully connected layer
    outputs = Dense(10, activation='softmax')(fused)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model