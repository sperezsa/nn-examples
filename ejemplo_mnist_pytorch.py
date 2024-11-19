# Importar la librería
import torch

# Y crear un tensor de forma manual
arreglo = [[2,3,4], [1,5,6]] # Arreglo 2D de 2 filas x 3 columnas
tensor1 = torch.tensor(arreglo)
print(tensor1)

tensor1.device

# Detectar la GPU
device = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Usando {device}")

tensor1 = tensor1.to(device)
print(tensor1.device)

print(tensor1.shape)

# Importar librerías requeridas
from torchvision import datasets # Para descargar el dataset
from torchvision.transforms import ToTensor # Para convertir los datos a Tensores
import matplotlib.pyplot as plt # Para graficar las imágenes + categorías

# Crear el directorio "datos" y ejecutar el siguiente código:
data_mnist = datasets.MNIST(
    root = "datos", # Carpeta donde se almacenará
    train=True, # True: 60.000 imágenes, False: 10.000 imágenes
    download=True,
    transform=ToTensor() # Convertir imágenes a tensores
)

figure = plt.figure(figsize=(8, 8))
fils, cols = 3, 3

for i in range(1, cols * fils + 1):
    # Escoger una imagen aleatoria
    sample_idx = torch.randint(len(data_mnist), size=(1,)).item()

    # Extraer imagen y categoría
    img, label = data_mnist[sample_idx]

    # Dibujar
    figure.add_subplot(fils, cols, i)
    plt.title(str(label)) # Categoría
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray") # Imagen
plt.show()

# Características de una imagen
print(f'Tipo de dato imagen: {type(img)}')
print(f'Tamaño imagen: {img.shape}')
print(f'Mínimo y máximo imagen: {img.min()}, {img.max()}')
print(f'Tipo de dato categoría: {type(label)}')

torch.manual_seed(123)

train, val, test = torch.utils.data.random_split(
    data_mnist, [0.8, 0.1, 0.1]
)

# Verificar tamaños
print(f'Tamaño set de entrenamiento: {len(train)}')
print(f'Tamaño set de validación: {len(val)}')
print(f'Tamaño set de prueba: {len(test)}')

# Y verificar el tipo de dato de train, val y test
print(f'Tipo de dato set "train": {type(train)}')
print(f'Tipo de dato set "val": {type(val)}')
print(f'Tipo de dato set "test": {type(test)}')


# Importar módulo nn
from torch import nn

"""
# Crear la Red Neuronal como una subclase de nn.Module
# Siempre se añaden dos métodos a esta subclase
# 1. Método "init": define la arquitectura de la red
# 2. Método "forward": define cómo será generada cada predicción

class RedNeuronal(nn.Module):
    # 1. Método "init"
    def __init__(self):
        super().__init__()

        # Y agregar secuencialmente las capas
        self.aplanar = nn.Flatten() # Aplanar imágenes de entrada
        self.red = nn.Sequential(
            nn.Linear(28*28, 15), # Capa de entrada + capa oculta
            nn.ReLU(), # Función de activación capa oculta
            nn.Linear(15,10), # Capa de salida SIN activación
        )

    # 2. Método "forward" (x = dato de entrada)
    def forward(self, x):
        # Definir secuencialmente las operaciones a aplicar
        x = self.aplanar(x) # Aplanar dato
        logits = self.red(x) # Generar predicción

        return logits


modelo = RedNeuronal().to(device)
print(modelo)

total_params = sum(p.numel() for p in modelo.parameters())
print("Número de parámetros a entrenar: ", total_params)
    
# Extraer una imagen y su categoría del set de entrenamiento
img, lbl = train[200]

print(type(img))
print(type(lbl))

# Convertir "lbl" a Tensor usando "tensor", definir tamaño igual a 1 (1 dato)
# con "reshape"
lbl = torch.tensor(lbl).reshape(1)
print(type(lbl))

img, lbl = img.to(device), lbl.to(device)

logits = modelo(img)
print(logits)


# Categoría predicha
y_pred = logits.argmax(1)

# Mostremos la imagen original
plt.imshow(img.cpu().squeeze(), cmap="gray");

# Y comparemos la categoría predicha con la categoría real
print(f'Logits: {logits}')
print(f'Categoría predicha: {y_pred[0]}')
print(f'Categoría real: {lbl[0]}')


# 0. Pérdida y optimizador
fn_perdida = nn.CrossEntropyLoss()
optimizador = torch.optim.SGD(modelo.parameters(), lr=0.2) # se ponen acá los parámetros para que se actualícen

# 1. Calcular pérdida
loss = fn_perdida(logits, lbl)
print(loss)

# 2. Calcular los gradientes de la pérdida
loss.backward()

# 3. Actualizar los parámetros del modelo
optimizador.step()
optimizador.zero_grad()

# img: dato, lbl: categoría real

# Propagación hacia adelante (generar predicciones)
logits = modelo(img)

# Propagación hacia atrás
loss = fn_perdida(logits, lbl) # Perdida
loss.backward() # Calcular gradientes
optimizador.step() # Actualizar parámetros del modelo
optimizador.zero_grad() # Borrar gradientes calculados anteriormente
"""
# Clase
class RedNeuronal(nn.Module):
    # 1. Método "init"
    def __init__(self):
        super().__init__()

        # Y agregar secuencialmente las capas
        self.aplanar = nn.Flatten() # Aplanar imágenes de entrada
        self.red = nn.Sequential(
            nn.Linear(28*28, 15), # Capa de entrada + capa oculta
            nn.ReLU(), # Función de activación capa oculta
            nn.Linear(15,10), # Capa de salida SIN activación
        )

    # 2. Método "forward" (x = dato de entrada)
    def forward(self, x):
        # Definir secuencialmente las operaciones a aplicar
        x = self.aplanar(x) # Aplanar dato
        logits = self.red(x) # Generar predicción

        return logits

# Instancia (llevada a la GPU)
modelo = RedNeuronal().to(device)

from torch.utils.data import DataLoader

# Definir el tamaño del lote
TAM_LOTE = 1000 # batch size

# Crear los "dataloaders" para los sets de entrenamiento y validación
train_loader = DataLoader(
    dataset=train,
    batch_size=TAM_LOTE,
    shuffle=True # Mezclar los datos aleatoriamente al crear cada lote
)

val_loader = DataLoader(
    dataset=val,
    batch_size=TAM_LOTE,
    shuffle=False
)


# Hiperparámetros
TASA_APRENDIZAJE = 0.1 # learning rate (0.1)
EPOCHS = 10 # Número de iteraciones de entrenamiento

# Función de pérdida y optimizador
fn_perdida = nn.CrossEntropyLoss()
optimizador = torch.optim.SGD(modelo.parameters(), lr=TASA_APRENDIZAJE)


def train_loop(dataloader, model, loss_fn, optimizer):
    # Cantidad de datos de entrenamiento y cantidad de lotes
    train_size = len(dataloader.dataset)
    nlotes = len(dataloader)

    # Indicarle a Pytorch que entrenaremos el modelo
    model.train()

    # Inicializar acumuladores pérdida y exactitud
    perdida_train, exactitud = 0, 0

    # Presentar los datos al modelo por lotes (de tamaño TAM_LOTE)
    for nlote, (X, y) in enumerate(dataloader):
        # Mover "X" y "y" a la GPU
        X, y = X.to(device), y.to(device)

        # Forward propagation
        logits = model(X)

        # Backpropagation
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Acumular valores de pérdida y exactitud
        # perdida_train <- perdida_train + perdida_actual
        # exactitud <- exactitud + numero_aciertos_actuales
        perdida_train += loss.item()
        exactitud += (logits.argmax(1)==y).type(torch.float).sum().item()

        # Imprimir en pantalla la evolución del entrenamiento (cada 10 lotes)
        if nlote % 10 == 0:
            # Obtener el valor de la pérdida (loss) y el número de datos procesados (ndatos)
            ndatos = nlote*TAM_LOTE

            # E imprimir en pantalla
            print(f"\tPérdida: {loss.item():>7f}  [{ndatos:>5d}/{train_size:>5d}]")

    # Al terminar de presentar todos los datos al modelo, promediar pérdida y exactitud
    perdida_train /= nlotes # Pérdida promedio = pérdida acumulada / número de lotes
    exactitud /= train_size # Exactitud promedio = exactitud acumulada / número de datos

    # E imprimir información
    print(f'\tExactitud/pérdida promedio:')
    print(f'\t\tEntrenamiento: {(100*exactitud):>0.1f}% / {perdida_train:>8f}')


def val_loop(dataloader, model, loss_fn):
    # Cantidad de datos de validación y cantidad de lotes
    val_size = len(dataloader.dataset)
    nlotes = len(dataloader)

    # Indicarle a Pytorch que validaremos el modelo
    model.eval()

    # Inicializar acumuladores pérdida y exactitud
    perdida_val, exactitud = 0, 0

    # Evaluar (generar predicciones) usando "no_grad"
    with torch.no_grad():
        for X, y in dataloader:
            # Mover "X" y "y" a la GPU
            X, y = X.to(device), y.to(device)

            # Propagación hacia adelante (predicciones)
            logits = model(X)

            # Acumular valores de pérdida y exactitud
            perdida_val += loss_fn(logits, y).item()
            exactitud += (logits.argmax(1) == y).type(torch.float).sum().item()

    # Tras generar las predicciones calcular promedios de pérdida y exactitud
    perdida_val /= nlotes
    exactitud /= val_size

    # E imprimir en pantalla
    print(f"\t\tValidación: {(100*exactitud):>0.1f}% / {perdida_val:>8f} \n")

"""
# Comentamos esto para que no vuelva a generar el modelo. Una vez guardado se puede emplear para hacer predicciones
for t in range(EPOCHS):
    print(f"Iteración {t+1}/{EPOCHS}\n-------------------------------")
    # Entrenar
    train_loop(train_loader, modelo, fn_perdida, optimizador)
    # Validar
    val_loop(val_loader, modelo, fn_perdida)
print("Listo, el modelo ha sido entrenado!") 

torch.save(modelo.state_dict(),'2024-04-25_model.pt')"""

# Cargamos el estado del diccionario del modelo
estado_diccionario = torch.load("2024-04-25_model.pt")
# Creamos un modelo vacío con la misma arquitectura
modelo = RedNeuronal().to(device)
# Cargamos el estado del diccionario en el modelo
modelo.load_state_dict(estado_diccionario)

def predecir(model, img):
    # Generar predicción
    logits = model(img)
    y_pred = logits.argmax(1).item()
    #print(logits)

    # Mostrar imagen original y categoría predicha
    plt.imshow(img.cpu().squeeze(), cmap="gray")
    plt.title(f'Categoría predicha: {y_pred}')
    plt.show()

# Tomar una imagen del set de prueba
img, lbl = test[200]
print(img)
print(lbl)

# Y generar la predicción
predecir(modelo, img)
