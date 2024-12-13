import torch 
import torch.nn as nn
from transformers import BertModel 

#Se construye la clase EdgeTransformer
#Se inicializa modelo base BERT
class EdgeTransformer(nn.Module):
    def __init__(self):
        super(EdgeTransformer, self).__init__()
        #Se carga el modelo BERT preentrenado para NLP
        self.bert=BertModel.from_pretrained('bert-base-uncased')


def forward(self, input_ids, attention_mask=None):
    #Pasan los datos a través del modelo BERT
    outputs=self.bert(input_ids=input_ids, attention_mask=attention_mask)
    #Se devuelve la última capa del modelo
    return outputs.last_hidden_state