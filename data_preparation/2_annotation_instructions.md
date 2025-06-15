# Guía de Anotación de Imágenes para el Detector de Personajes de Star Wars

## 📋 Introducción

Este documento proporciona instrucciones detalladas para anotar las imágenes de personajes de Star Wars que serán utilizadas para entrenar nuestro modelo de detección de objetos. La anotación es un paso crucial en el proceso de entrenamiento y afectará directamente la precisión de nuestro modelo.

## 🎯 Personajes a Anotar

1. Darth Vader
2. Luke Skywalker
3. Yoda
4. R2-D2
5. C-3PO
6. Chewbacca
7. Han Solo
8. Leia Organa

## 🛠️ Herramientas Recomendadas

### Opción 1: LabelImg (Recomendada para principiantes)

1. **Instalación**:
   ```bash
   pip install labelImg
   ```

2. **Ejecución**:
   ```bash
   labelImg
   ```

### Opción 2: Roboflow (Recomendada para equipos)

1. Visita [Roboflow](https://roboflow.com)
2. Crea una cuenta gratuita
3. Crea un nuevo proyecto
4. Sube las imágenes

## 📝 Instrucciones de Anotación

### Usando LabelImg

1. **Configuración Inicial**:
   - Abre LabelImg
   - Ve a View > Auto Save
   - Ve a Change Save Dir y selecciona la carpeta donde guardarás las anotaciones
   - En la barra lateral derecha, selecciona "YOLO" como formato de guardado

2. **Proceso de Anotación**:
   - Abre una imagen (File > Open Dir)
   - Para cada personaje en la imagen:
     1. Presiona 'W' para crear un nuevo bounding box
     2. Dibuja el rectángulo alrededor del personaje
     3. Selecciona la clase correcta del menú desplegable
     4. Presiona Enter para guardar la anotación
   - Usa 'D' para ir a la siguiente imagen
   - Usa 'A' para ir a la imagen anterior

### Usando Roboflow

1. **Subida de Imágenes**:
   - Arrastra y suelta la carpeta `dataset_raw` en la interfaz de Roboflow
   - Selecciona "Object Detection" como tipo de proyecto

2. **Creación de Clases**:
   - Crea las 8 clases correspondientes a los personajes
   - Asegúrate de que los nombres coincidan exactamente con la lista proporcionada

3. **Anotación**:
   - Usa la herramienta de dibujo para crear bounding boxes
   - Asigna la clase correcta a cada anotación
   - Guarda cada anotación

## ⚠️ Reglas Importantes

1. **Precisión**:
   - El bounding box debe incluir TODO el personaje
   - Incluye accesorios característicos (ej. sable de luz, blaster)
   - No incluyas otros personajes en el mismo bounding box

2. **Consistencia**:
   - Mantén un estilo consistente en todas las anotaciones
   - Usa el mismo nivel de detalle para todos los personajes

3. **Calidad**:
   - Anota SOLO personajes claramente visibles
   - Ignora imágenes borrosas o de baja calidad
   - No anotes personajes parcialmente visibles

## 📊 Estructura de Archivos

```
dataset_raw/
├── Darth_Vader/
│   ├── Darth_Vader_1.jpg
│   ├── Darth_Vader_1.txt
│   └── ...
├── Luke_Skywalker/
│   ├── Luke_Skywalker_1.jpg
│   ├── Luke_Skywalker_1.txt
│   └── ...
└── ...
```

## 🔍 Verificación de Calidad

Antes de considerar completada la anotación, verifica:

1. Cada imagen tiene su archivo .txt correspondiente
2. Los archivos .txt contienen el formato YOLO correcto:
   ```
   <class_id> <x_center> <y_center> <width> <height>
   ```
3. No hay imágenes sin anotar
4. No hay anotaciones duplicadas
5. Los bounding boxes son precisos y consistentes

## 🚀 Siguiente Paso

Una vez completada la anotación, ejecuta el script `3_prepare_dataset.py` para preparar el dataset para el entrenamiento:

```bash
python data_preparation/3_prepare_dataset.py
```

## ❓ Soporte

Si encuentras problemas durante la anotación:

1. Revisa la [documentación de LabelImg](https://github.com/tzutalin/labelImg)
2. Consulta la [documentación de Roboflow](https://docs.roboflow.com)
3. Abre un issue en el repositorio del proyecto 