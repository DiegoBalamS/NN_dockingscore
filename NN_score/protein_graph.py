import pandas as pd

def leer_pdb(archivo_pdb):
    with open(archivo_pdb, 'r') as file:
        lineas = file.readlines()

    datos_atomos = []
    for linea in lineas:
        if linea.startswith("ATOM"):
            elementos = linea.split()
            tipo_atomo = elementos[2]
            coordenadas = [float(elementos[5]), float(elementos[6]), float(elementos[7])]
            datos_atomos.append((tipo_atomo, coordenadas))

    return pd.DataFrame(datos_atomos, columns=['Tipo', 'Coordenadas'])

datos_pdb = leer_pdb('ruta_a_tu_archivo_pdb')
print(datos_pdb.head(10))

"""
Este código leerá un archivo PDB y extraerá el tipo de átomo y sus coordenadas, almacenándolos en un DataFrame de Pandas para su posterior procesamiento. 

8. **Manejo de Información Adicional**:
- Además de las coordenadas y tipos de átomos, puedes necesitar manejar información adicional como factores de ocupación y temperaturas, si estos son relevantes para tu análisis.

9. **Automatización y Validación**:
- Implementa un proceso automatizado que pueda manejar múltiples archivos PDB.
- Valida que la extracción y procesamiento de los datos se realice correctamente, especialmente si vas a trabajar con un gran conjunto de datos.

10. **Integración con SchNet**:
 - Una vez que tengas tus datos en el formato correcto, puedes empezar a integrarlos con SchNet para entrenar tu modelo o realizar predicciones.

Este proceso puede variar en complejidad dependiendo de la especificidad de tus necesidades y los detalles de tu proyecto. Asegúrate de ajustar estos pasos según lo que requiera tu investigación o aplicación específica.
"""
