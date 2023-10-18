# Glosario

- **Cápsula**: Fragmento de división de la app. Es en torno a 1000 caracteres
- **Capítulo**: Fragmento de división de audio. Se forman para que sean 25 en cada libro


## Files
- `00 Audiobook`: 
- `01 Preprocess`
- `02 Preprocess summary`
- `03 Audiobook`

# 1. Libros para Cuéntame

0. Cargar los libros en `Calibre` (unos 20)
   a. Ordenarlos por fecha, y convertirlos a txt
   b. Buscarles las carátulas y actualizar los metadatos    
1. Ejecutar el notebook `01 Preprocess` para 
   a. Con la opción `b` se muestran los recién cargados (la última fecha)

# 2. Audiobooks

Para creación se audiobooks en mp3
1. Transformar a txt en **Calibre**
2. Ejecutar el notebook `01 Preprocess` para 
   1. Seleccionar libros de manera aleatoria de una carpeta
   1. Crear el summary
   2. Los números de las cápsulas donde cortar (No hace falta cortar)
4. Ejecutar del notebook `02 Preprocess_summary` para obtener la carátula 
5. Ejecutar `03 Audiobook`

# TODO:
- quitar lo de los dicionarios (especialemente en summari.json)
- que summary sea un yaml
- por qué se pone feo el nombre del libroo en summary
   - quitar los paréntesis
- entender por qué guardo "names" en summary
- guardar las cápsulas en yaml separado por libro, para poder usar como audiobook
- hacer con Azure

# nota:
 para subir mas rápido as capsulas lo hacemos ppor mongo (batch), por eso hay que borrarlas por mongo