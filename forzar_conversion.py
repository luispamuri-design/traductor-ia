import os
import sys
import json
from types import ModuleType

# 1. Parches de inicio obligatorios para Windows
jax_mock = ModuleType('jax')
jax_experimental_mock = ModuleType('jax.experimental')
jax_experimental_mock.jax2tf = lambda *args, **kwargs: None
sys.modules['tensorflow_decision_forests'] = ModuleType('tensorflow_decision_forests')
sys.modules['jax'] = jax_mock
sys.modules['jax.experimental'] = jax_experimental_mock

import tensorflow as tf
from tensorflowjs.converters import save_keras_model
import h5py

ruta_h5 = "modelo_senas.h5"
carpeta_salida = "static/modelo_web"

print("=== Iniciando Convertidor por Cirugia de Archivo H5 ===")

if not os.path.exists(ruta_h5):
    print(f"Error: No encuentro el archivo {ruta_h5} en esta carpeta.")
    exit()

try:
    print("Abriendo archivo .h5 para cirugia estructural...")
    with h5py.File(ruta_h5, 'r+') as f:
        if 'model_config' in f.attrs:
            print("Limpiando la configuracion de las capas...")
            config_str = f.attrs['model_config']
            if isinstance(config_str, bytes):
                config_str = config_str.decode('utf-8')
            
            model_json = json.loads(config_str)
            
            # Buscamos y limpiamos recursivamente TODO lo que rompa a Keras 2
            def limpiar_diccionario_keras(d):
                if isinstance(d, dict):
                    # Parche para InputLayer
                    if d.get('class_name') == 'InputLayer' or d.get('class_name') == 'Input':
                        cfg = d.get('config', {})
                        cfg.pop('optional', None)
                        if 'batch_shape' in cfg:
                            cfg['input_shape'] = cfg.pop('batch_shape')[1:]
                    
                    # Parche general para cualquier otra capa
                    cfg = d.get('config', {})
                    if isinstance(cfg, dict):
                        cfg.pop('quantization_config', None)
                        cfg.pop('optional', None)
                        if 'batch_shape' in cfg:
                            cfg['input_shape'] = cfg.pop('batch_shape')[1:]
                        if 'dtype' in cfg and isinstance(cfg['dtype'], dict):
                            cfg['dtype'] = cfg['dtype'].get('config', {}).get('name', 'float32')
                    
                    # === PARCHE PARA EL OPTIMIZADOR ===
                    # Si encontramos la configuración del optimizador, removemos weight_decay
                    if 'optimizer' in d or 'optimizer_config' in d:
                        opt = d.get('optimizer', d.get('optimizer_config', {}))
                        if isinstance(opt, dict) and 'config' in opt:
                            opt['config'].pop('weight_decay', None)
                    
                    for k, v in d.items():
                        limpiar_diccionario_keras(v)
                elif isinstance(d, list):
                    for item in d:
                        limpiar_diccionario_keras(item)

            limpiar_diccionario_keras(model_json)
            f.attrs['model_config'] = json.dumps(model_json).encode('utf-8')
            
        # === PARCHE EXTRA: Por si la configuración del entrenamiento está por separado ===
        if 'training_config' in f.attrs:
            print("Limpiando la configuracion del entrenamiento y optimizador...")
            train_str = f.attrs['training_config']
            if isinstance(train_str, bytes):
                train_str = train_str.decode('utf-8')
            train_json = json.loads(train_str)
            
            # Quitamos el weight_decay de la sección de entrenamiento
            if isinstance(train_json, dict) and 'optimizer_config' in train_json:
                opt_cfg = train_json['optimizer_config'].get('config', {})
                if isinstance(opt_cfg, dict):
                    opt_cfg.pop('weight_decay', None)
            
            f.attrs['training_config'] = json.dumps(train_json).encode('utf-8')
            
        print("¡Cirugia exitosa! Los atributos incompatibles han sido eliminados del archivo.")

    # 2. Carga tradicional en TensorFlow (compilar=False ignora los problemas del optimizador al cargar)
    print("\nCargando el modelo reparado en memoria (sin compilar optimizador)...")
    modelo = tf.keras.models.load_model(ruta_h5, compile=False)
    print("¡Modelo cargado con exito en memoria!")
    
    print("\nExportando a formato TensorFlow.js...")
    save_keras_model(modelo, carpeta_salida)
    
    print(f"\n¡LOGRADO! Tu modelo listo para la web esta en: {carpeta_salida}")
except Exception as e:
    print(f"\nOcurrio un error inesperado: {e}")