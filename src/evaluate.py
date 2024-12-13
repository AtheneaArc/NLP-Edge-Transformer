import torch
from edge_transformer import EdgeTransformer
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import accuracy_score

#Cargar el conjunto de datos de prueba
dataset=load_dataset('imdb', split='test')
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_data(dataset):
    return tokenizer(dataset['text'], padding=True, truncation=True, return_tensors='pt')

#Preprocesar los datos
tokenized_dataset=dataset.map(preprocess_data, batched=True)
test_loader=DataLoader(tokenized_dataset, batch_size=8)

#Cargar el modelo entrenado
model=EdgeTransformer()
model.load_state_dict(torch.load('best_model.pth'))

#Evaluar el modelo
model.eval()
predictions, labels = [],[]

for batch in test_loader:
    input_ids=batch['input_ids']
    attention_mask=batch['attention_mask']
    labels_batch=batch['label']

    with torch.no_grad():
        outputs=model(input_ids,attention_mask)
        _, predicted=torch.max(outputs, dim=1)

    predictions.extend(predicted.cpu().numpy())
    labels.extend(labels_batch.cpu().numpy())    

accuracy=accuracy_score(labels, predictions)
print(f'Accuracy: {accuracy}')