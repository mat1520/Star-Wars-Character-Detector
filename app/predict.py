from ultralytics import YOLO
import cv2
import os

def predict_image(image_path):
    # Limpiar la ruta de comillas si existen
    image_path = image_path.strip('"').strip("'")
    
    # Verificar si el archivo existe
    if not os.path.exists(image_path):
        print(f"Error: No se puede encontrar el archivo en la ruta: {image_path}")
        return
    
    # Cargar el modelo entrenado
    model = YOLO('runs/detect/star_wars_detector/weights/best.pt')
    
    # Realizar la predicción
    results = model(image_path)
    
    # Obtener la imagen con las predicciones
    result = results[0]
    img_with_boxes = result.plot()
    
    # Guardar la imagen con las predicciones
    output_path = 'prediction_result.jpg'
    cv2.imwrite(output_path, img_with_boxes)
    
    # Imprimir las predicciones
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names[class_id]
        print(f'Detectado: {class_name} con confianza: {confidence:.2f}')
    
    print(f'\nImagen con predicciones guardada como: {output_path}')

if __name__ == '__main__':
    print("Por favor, ingresa la ruta de la imagen a predecir.")
    print("Ejemplo: C:\\Users\\Ariel\\Downloads\\imagen.jpg")
    image_path = input('Ruta de la imagen: ')
    predict_image(image_path) 