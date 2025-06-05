import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('outputs/models/LSTM_regression/dl_regression_model_20250426_033855.keras')

# Print the model summary to see all layers and parameters
model.summary()

# For inspecting each layer properly
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name}, Type: {type(layer).__name__}")
    
    # For input shape, need to check if the attribute exists
    # or access through get_config() for complex layers
    try:
        # Try to get input shape from the layer's input tensor
        if hasattr(layer, 'input') and layer.input is not None:
            print(f"  Input shape: {layer.input.shape}")
        else:
            # For some layers like Bidirectional, we need to get info from config
            config = layer.get_config()
            print(f"  Layer config: {config.get('layer', {}).get('config', {}).get('batch_input_shape', 'Not available')}")
    except:
        print("  Input shape: Not directly accessible")
    
    # Similarly for output shape
    try:
        if hasattr(layer, 'output') and layer.output is not None:
            print(f"  Output shape: {layer.output.shape}")
        else:
            print("  Output shape: See model summary")
    except:
        print("  Output shape: Not directly accessible")
    
    print(f"  Parameters: {layer.count_params()}")
    
    # Special handling for Bidirectional layers
    if isinstance(layer, tf.keras.layers.Bidirectional):
        print(f"  Forward layer: {layer.forward_layer.__class__.__name__}")
        print(f"  Backward layer: {layer.backward_layer.__class__.__name__}")
        if hasattr(layer.forward_layer, 'units'):
            print(f"  Units: {layer.forward_layer.units}")


