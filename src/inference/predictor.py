"""
Inference utilities for BERT models
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging
from transformers import BertTokenizer

from ..models.classifier import BERTForClassification
from ..data.dataset import AGNewsDataset


logger = logging.getLogger(__name__)


class NewsClassifier:
    """News classifier using BERT"""
    
    def __init__(
        self, 
        model_path: str, 
        tokenizer_name: str = "bert-base-uncased",
        device: str = None,
        seq_len: int = 128
    ):
        """
        Initialize news classifier.
        
        Args:
            model_path: Path to saved model
            tokenizer_name: Tokenizer to use
            device: Device to run inference on
            seq_len: Maximum sequence length
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_len = seq_len
        self.labels = ["World", "Sports", "Business", "Sci/Tech"]
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        
        # Initialize dataset processor
        self.dataset = AGNewsDataset(tokenizer=self.tokenizer, seq_len=seq_len)
        
        # Load model
        self.model = None
        self.load_model(model_path)
        
    def load_model(self, model_path: str):
        """Load the trained model"""
        try:
            logger.info(f"Loading model from {model_path}")
            
            # This is a simplified loading - you'd need to adapt based on your actual model saving format
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Initialize model architecture (you'd need to adapt this)
            from ..models.bert_model import BERT
            bert_encoder = BERT(vocab_size=len(self.tokenizer.vocab), d_model=768)
            self.model = BERTForClassification(bert_encoder, num_classes=4)
            
            # Load state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, text: str) -> Dict[str, any]:
        """
        Predict category for a single text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with prediction results
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
            
        try:
            # Process text
            processed = self.dataset.from_text(text)
            
            # Move to device and add batch dimension
            input_data = {
                k: v.unsqueeze(0).to(self.device) 
                for k, v in processed.items() 
                if k in ['bert_input', 'segment_label']
            }
            
            # Inference
            with torch.no_grad():
                logits = self.model(input_data["bert_input"], input_data["segment_label"])
                probabilities = F.softmax(logits, dim=1)
                pred_class = torch.argmax(logits, dim=1).item()
                confidence = probabilities[0, pred_class].item()
            
            return {
                'category': self.labels[pred_class],
                'confidence': confidence,
                'class_id': pred_class,
                'all_probabilities': {
                    label: prob.item() 
                    for label, prob in zip(self.labels, probabilities[0])
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction failed for text: {text[:50]}... Error: {e}")
            raise

    def predict_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Predict categories for multiple texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of prediction results
        """
        results = []
        
        for text in texts:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to predict for text: {text[:50]}... Error: {e}")
                results.append({
                    'category': 'Unknown',
                    'confidence': 0.0,
                    'class_id': -1,
                    'error': str(e)
                })
        
        return results

    def get_top_k_predictions(self, text: str, k: int = 2) -> List[Tuple[str, float]]:
        """
        Get top-k predictions for a text.
        
        Args:
            text: Input text
            k: Number of top predictions to return
            
        Returns:
            List of (category, confidence) tuples
        """
        result = self.predict(text)
        all_probs = result['all_probabilities']
        
        # Sort by confidence
        sorted_predictions = sorted(
            all_probs.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_predictions[:k]


class BatchNewsClassifier:
    """Optimized classifier for batch processing"""
    
    def __init__(self, model_path: str, batch_size: int = 32):
        """
        Initialize batch classifier.
        
        Args:
            model_path: Path to saved model
            batch_size: Batch size for processing
        """
        self.classifier = NewsClassifier(model_path)
        self.batch_size = batch_size

    def predict_large_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Efficiently process large batches of texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of prediction results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_results = self.classifier.predict_batch(batch_texts)
            results.extend(batch_results)
            
            if i % (self.batch_size * 10) == 0:
                logger.info(f"Processed {i + len(batch_texts)}/{len(texts)} texts")
        
        return results
