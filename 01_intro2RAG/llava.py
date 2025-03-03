import os
import pydicom
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaTokenizer, LlamaForSequenceClassification
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

class DicomCoronaryDataset(Dataset):
    def __init__(self, dicom_dir, annotations, tokenizer, transform=None):
        """
        Args:
            dicom_dir (str): Directory with DICOM images
            annotations (dict): Dictionary with image_id: {segment: str, stenosis: float}
            tokenizer: LLaMA tokenizer
            transform: Image transformations
        """
        self.dicom_dir = dicom_dir
        self.annotations = annotations
        self.image_ids = list(annotations.keys())
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load DICOM image
        dicom_path = os.path.join(self.dicom_dir, f"{image_id}.dcm")
        dicom = pydicom.dcmread(dicom_path)
        image = dicom.pixel_array
        
        # Convert to PIL Image and apply transformations
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        
        # Get annotations
        segment = self.annotations[image_id]['segment']
        stenosis = self.annotations[image_id]['stenosis']
        
        # Tokenize text
        text = f"Coronary segment: {segment}, Stenosis: {stenosis}%"
        encoding = self.tokenizer(text, 
                                padding='max_length',
                                max_length=128,
                                truncation=True,
                                return_tensors='pt')
        
        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'segment': segment,
            'stenosis': torch.tensor(stenosis, dtype=torch.float)
        }

class CoronaryLlamaModel(nn.Module):
    def __init__(self, num_segments, pretrained_model_name):
        super().__init__()
        
        # Image feature extractor (ResNet)
        self.image_encoder = torch.hub.load('pytorch/vision:v0.10.0', 
                                          'resnet50', 
                                          pretrained=True)
        self.image_encoder.fc = nn.Linear(2048, 768)  # Match LLaMA hidden size
        
        # LLaMA model
        self.llama = LlamaForSequenceClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_segments
        )
        
        # Stenosis regression head
        self.stenosis_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, images, input_ids, attention_mask):
        # Extract image features
        image_features = self.image_encoder(images)
        
        # Get LLaMA outputs
        llama_outputs = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Combine image and text features
        combined_features = image_features + llama_outputs.hidden_states[-1][:, 0, :]
        
        # Segment classification
        segment_logits = self.llama.classifier(combined_features)
        
        # Stenosis regression
        stenosis_pred = self.stenosis_head(combined_features)
        
        return segment_logits, stenosis_pred

def train_model():
    # Initialize tokenizer and model
    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b')
    model = CoronaryLlamaModel(
        num_segments=15,  # Number of coronary segments
        pretrained_model_name='meta-llama/Llama-2-7b'
    )
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    dataset = DicomCoronaryDataset(
        dicom_dir='path/to/dicom/images',
        annotations=load_annotations(),  # You need to implement this
        tokenizer=tokenizer,
        transform=transform
    )
    
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss functions
    segment_criterion = nn.CrossEntropyLoss()
    stenosis_criterion = nn.MSELoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            segment_labels = batch['segment'].to(device)
            stenosis_labels = batch['stenosis'].to(device)
            
            optimizer.zero_grad()
            
            segment_logits, stenosis_pred = model(images, input_ids, attention_mask)
            
            # Calculate losses
            segment_loss = segment_criterion(segment_logits, segment_labels)
            stenosis_loss = stenosis_criterion(stenosis_pred.squeeze(), stenosis_labels)
            
            total_loss = segment_loss + stenosis_loss
            total_loss.backward()
            
            optimizer.step()
            
        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                segment_labels = batch['segment'].to(device)
                stenosis_labels = batch['stenosis'].to(device)
                
                segment_logits, stenosis_pred = model(images, input_ids, attention_mask)
                
                segment_loss = segment_criterion(segment_logits, segment_labels)
                stenosis_loss = stenosis_criterion(stenosis_pred.squeeze(), stenosis_labels)
                
                val_loss += (segment_loss + stenosis_loss).item()
                
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss/len(val_loader):.4f}")

def predict(model, image_path, tokenizer):
    """
    Make predictions on a new DICOM image
    """
    # Load and preprocess image
    dicom = pydicom.dcmread(image_path)
    image = dicom.pixel_array
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    image = transform(Image.fromarray(image)).unsqueeze(0)
    
    # Prepare text input
    text = "Analyze coronary segments and stenosis"
    encoding = tokenizer(text, 
                        padding='max_length',
                        max_length=128,
                        truncation=True,
                        return_tensors='pt')
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        segment_logits, stenosis_pred = model(
            image,
            encoding['input_ids'],
            encoding['attention_mask']
        )
    
    # Process predictions
    segment_pred = torch.argmax(segment_logits, dim=1)
    stenosis_value = stenosis_pred.item()
    
    return {
        'segment': segment_pred.item(),
        'stenosis_percentage': stenosis_value
    }

if __name__ == "__main__":
    train_model()