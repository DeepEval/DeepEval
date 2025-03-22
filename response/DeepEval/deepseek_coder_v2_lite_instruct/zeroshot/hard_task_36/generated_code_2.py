import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main pathway
    main_pathway = Conv2D(32, (3, 3), activation='relu')(input_layer)
    main_pathway = Conv2D(32, (1, 1), activation='relu')(main_pathway)
    main_pathway = Conv2D(32, (1, 1), activation='relu')(main_pathway)
    main_pathway = MaxPooling2D(pool_size=(2, 2))(main_pathway)
    main_pathway = Dropout(0.5)(main_pathway)
    
    # Branch pathway
    branch_pathway = Conv2D(32, (3, 3), activation='relu')(main_pathway)
    
    # Fused pathway
    fused_pathway = Add()([main_pathway, branch_pathway])
    
    # Global average pooling
    global_avg_pool = GlobalAveragePooling2D()(fused_pathway)
    
    # Flatten layer
    flattened = Flatten()(global_avg_pool)
    
    # Fully connected layer
    output_layer = Dense(10, activation='softmax')(flattened)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage
model = dl_model()
model.summary()