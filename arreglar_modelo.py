import h5py

# Abrimos el archivo original en modo lectura/escritura
with h5py.File("modelo_senas.h5", "r+") as f:
    # Revisamos si el modelo tiene la configuración guardada
    if 'model_config' in f.attrs:
        config = f.attrs['model_config']
        
        # El archivo guarda la configuración como texto (bytes)
        if isinstance(config, bytes):
            config = config.decode('utf-8')
            
        print("Corrigiendo incompatibilidades de Keras 3 a Keras 2...")
        
        # Reemplazamos los términos modernos por los que entiende Keras 2
        config = config.replace('"batch_shape"', '"batch_input_shape"')
        config = config.replace(', "optional": false', '')
        config = config.replace(', "optional": true', '')
        
        # Volvemos a guardar la configuración corregida
        f.attrs['model_config'] = config.encode('utf-8')
        print("¡Modelo modificado con éxito!")
    else:
        print("No se encontró la configuración del modelo o ya fue modificado.")