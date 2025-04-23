
# Importing the libraries needed
import pandas as pd
import torch
import time
import gc
import os
import random
import numpy as np
import transformers
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer, BertModel, BertTokenizer #, AlbertModel, AlbertTokenizer
import warnings

warnings.filterwarnings('ignore')

# Get the absolute path of the current script
current_dir = os.path.abspath(os.path.dirname(__file__))
# Change the working directory to the directory where the current script is located
os.chdir(current_dir)


# Setting up the device for GPU usage

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print('GPU ',cuda.is_available())


# clean memory
gc.collect()
torch.cuda.empty_cache()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random seed setted.")

seed = 42
set_seed(seed)


# Import the csv into pandas dataframe 
def load_data(dataset):
    train_df = pd.read_csv(os.path.join(current_dir,f'dataset/{dataset}/train.csv'))
    test_df= pd.read_csv(os.path.join(current_dir,f'dataset/{dataset}/test.csv'))
    return train_df, test_df



class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        text = str(self.data.text[index])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.label[index], dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len



# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 

class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistillBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 1)
    
    def forward(self, ids, mask):
        output_1= self.l1(ids, mask)
        output_2 = self.l2(output_1[0])
        output = self.l3(output_2)
        return output



class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 1)
    
    def forward(self, ids, mask):
        output_1= self.l1(ids, mask)
        output_2 = self.l2(output_1[0])
        output = self.l3(output_2)
        return output
'''
class ALBERTClass(torch.nn.Module):
    def __init__(self):
        super(ALBERTClass, self).__init__()
        self.l1 = AlbertModel.from_pretrained("albert/albert-base-v2")
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 1)
    
    def forward(self, ids, mask):
        output_1= self.l1(ids, mask)
        output_2 = self.l2(output_1[0])
        output = self.l3(output_2)
        return output
'''
def compute_metrics(preds, targets):
    return f1_score(targets, preds, average='macro')

def train(model, training_loader, loss_function, optimizer, device):
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask).squeeze()

        optimizer.zero_grad()
        loss = loss_function(outputs, targets)
        if _%5000==0:
            print(f' Loss:  {loss.item()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, testing_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            outputs = model(ids, mask).squeeze()
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    return compute_metrics(all_preds, all_targets)


datasets = ['sst2']#'AG_News','CNN_News',
models = {
    'DistilBERT': (DistilBertTokenizer.from_pretrained('distilbert-base-uncased'), DistillBERTClass),
    #'BERT': (BertTokenizer.from_pretrained('bert-base-uncased'), BERTClass),
    #'ALBERT': (AlbertTokenizer.from_pretrained("albert/albert-base-v2"), ALBERTClass)
}
max_len = 256
batch_size = 32
epochs=1
results = []
learning_rate=5e-5

for model_name, (tokenizer, model_class) in models.items():
    for dataset in datasets:
        print(f'-------------Training {model_name} for {dataset}-----------')
        train_df,test_df = load_data(dataset)
         
        num_labels = len(train_df['label'].unique())

        train_dataset = TextDataset(train_df, tokenizer, max_len)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_dataset = TextDataset(test_df, tokenizer, max_len)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        model = model_class().to(device)
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

        start_time = time.time()
        for epoch in range(epochs):
            print(f'Trainning epoch: {epoch+1} ')
            train(model, train_loader, loss_function, optimizer, device)
        train_time = time.time() - start_time

        start_time = time.time()
        test_macro_f1 = evaluate(model, test_loader, device)
        test_time = time.time() - start_time

        total_time = train_time + test_time
        print({
            'Model Name': model_name,
            'Dataset': dataset,
            'Macro-F1': test_macro_f1,
            'Running Time': total_time
        })
        
        results.append({
            'Model Name': model_name,
            'Dataset': dataset,
            'Macro-F1': test_macro_f1,
            'Running Time': total_time
        })
        gc.collect()
        torch.cuda.empty_cache()

# save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('bert_results.csv', index=False)

