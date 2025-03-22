from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Add
from keras.models import Model

def dl_model():
    # Main path
    main_path = Conv2D(32, (3, 3), activation='relu')(input_layer)
    main_path = MaxPooling2D((2, 2))(main_path)
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = Dropout(0.25)(main_path)
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = MaxPooling2D((2, 2))(main_path)
    main_path = Flatten()(main_path)
    main_path = Dense(128, activation='relu')(main_path)
    main_path = Dropout(0.5)(main_path)
    main_path = Dense(10, activation='softmax')(main_path)
    
    # Branch path
    branch_path = Conv2D(32, (3, 3), activation='relu')(input_layer)
    branch_path = MaxPooling2D((2, 2))(branch_path)
    branch_path = Conv2D(64, (3, 3), activation='relu')(branch_path)
    branch_path = Dropout(0.25)(branch_path)
    branch_path = Conv2D(64, (3, 3), activation='relu')(branch_path)
    branch_path = MaxPooling2D((2, 2))(branch_path)
    branch_path = Flatten()(branch_path)
    branch_path = Dense(128, activation='relu')(branch_path)
    branch_path = Dropout(0.5)(branch_path)
    branch_path = Dense(10, activation='softmax')(branch_path)
    
    # Combine the outputs of the main and branch paths
    combined_path = Add()([main_path, branch_path])
    
    # Define the model
    model = Model(inputs=input_layer, outputs=combined_path)
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model