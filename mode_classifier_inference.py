import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np

class VisualModeClassifier:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        self.model = self._load_model(checkpoint_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.buffer = []
        self.buffer_size = 5 # Smoothing window

    def _load_model(self, path):
        print(f"Loading Mode Classifier from {path}...")
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)
        
        # Load weights
        try:
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading classifier weights: {e}")
            print("Warning: Visual Classifier is initializing with random weights (for testing only)")
        
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_np):
        """
        Args:
            image_np: (H, W, 3) numpy array (uint8), e.g. from ts.observation['images']['top']
        Returns:
            predicted_mode (int): 0 (Indep) or 1 (Coop)
            confidence (float): Probability of the predicted class
        """
        try:
            img_pil = Image.fromarray(image_np.astype('uint8'))
            img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
                # prob[0] is Indep, prob[1] is Coop
                prob_coop = probs[0][1].item()
                pred_label = 1 if prob_coop > 0.5 else 0
                confidence = prob_coop if pred_label == 1 else (1.0 - prob_coop)
                
                # Smoothing
                self.buffer.append(pred_label)
                if len(self.buffer) > self.buffer_size:
                    self.buffer.pop(0)
                
                # Majority vote
                avg_label = sum(self.buffer) / len(self.buffer)
                smoothed_pred = 1 if avg_label > 0.5 else 0
                
                return smoothed_pred, confidence
                
        except Exception as e:
            print(f"Mode Classifier Inference Error: {e}")
            return 0, 0.0 # Default to Indep safely?
