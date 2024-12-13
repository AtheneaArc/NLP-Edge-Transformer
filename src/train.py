import torch
#Import modelo EdgeTransformer
from edge_transformer import EdgeTransformer
#Import tokenizador de BERT
from transformers import BertTokenizer
#DataLoader para cargar los datos
from torch.utils.data import DataLoader
#Librería dataset de HuggingFace
from datasets import load_dataset

#Subconjunto del dataset 'imdb' para sentiment-clasification
dataset=load_dataset('imdb', split='train[:10%]')
print(f"Dataset cargado: {len(dataset)} muestras")

#Tokenizador de BERT, convierte texto en identificador de tokens
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')

#Función para preprocesar datos que el tokenizador de BERT entiende
def preprocess_data(dataset):
    return tokenizer(dataset['text'], padding=True, truncation=True, return_tensors='pt')

#Se aplica el preprocess a todo el dataset

tokenized_dataset=dataset.map(preprocess_data, batched=True)

#Crea un DataLoader para iterar sobre los datos en batch(lotes)
train_loader=DataLoader(tokenized_dataset, batch_size=8)

#Inicia modelo EdgeTransformer
model=EdgeTransformer

#Optimizador (AdamW, versión de optimizador Adam que funciona mejor con transformers)
optimizer=torch.optim.Adam(model.árameters(), lr=1e-5)

#Función de pérdida (CrossEntropyLoss, se utiliza para clasificación)
loss_fn=torch.nn.CrossEntropyLoss()

#Train
model.train() #modo train
for batch in train_loader:
    #Limpiar gradientes de la iteración anterior
    optimizer.zero_grad()
    #Extraer input y labels del batch
    input_ids=batch['input_ids']
    attention_mask=batch['attention_mask']
    labels=batch['label']
    #Forward del modelo
    outputs=model(input_ids, attention_mask)
    #Calcular loss entre predicciones del modelo y los labels reales
    loss=loss_fn(outputs,labels)
    #Calcular los gradientes (retroprogramación)
    loss.backward()
    #Actualización de pesos del modelo
    optimizer.step()
    #Print de loss para monitorear el train
    print(f'Loss: {loss.item()}')


