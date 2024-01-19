from enum import IntEnum

import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer, RobertaForSequenceClassification


class ModelType(IntEnum):
    AUTO_MODEL = 1
    ROBERTA_SEQUENCE_MODEL = 2
    SPANBERT_MDOEL = 3


class ModelBaseClass(nn.Module):
    def __init__(self, base_model_name: str, no_classes: int, model_type: ModelType):
        super(ModelBaseClass, self).__init__()
        self.base_model_name = base_model_name
        self.no_classes = no_classes
        self.model = None
        
        if model_type == ModelType.AUTO_MODEL or model_type == ModelType.SPANBERT_MDOEL:
            self.model = AutoModel.from_pretrained(base_model_name)
        elif model_type == ModelType.ROBERTA_SEQUENCE_MODEL:
            self.model = RobertaForSequenceClassification.from_pretrained(base_model_name)
            
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
    def tokenizer(self):
        return self._tokenizer
    
    def add_special_token_to_tokenizer(self, token):
        if token not in self._tokenizer.special_tokens_map:
            self._tokenizer.add_tokens(token, special_tokens=True)
            self.model.resize_token_embeddings(len(self._tokenizer))
    
    def forward(self, x):
        raise NotImplementedError(f'{self.__class__.__name__} did not implement forward.')
        
    def freeze_base_model(self):
        for param in self.model.parameters():
            param.requires_grad = False
            
    def unfreeze_base_model(self):
        for param in self.model.parameters():
             param.requires_grad = True
        
    def _span_loss(self, x, span_logits):
        total_loss = None
        
        start_logits, end_logits = span_logits[:, 0].contiguous(), span_logits[:, 1].contiguous()
        
        start_pos = x.get('start_positions', None)
        end_pos = x.get('end_positions', None)
        
        if start_pos is not None and end_pos is not None:
            if len(start_pos.size()) > 1:
                start_pos = start_pos.squeeze(-1)
            if len(end_pos.size()) > 1:
                end_pos = end_pos.squeeze(-1)
                
            ignored_idx = start_logits.size(1)
            start_pos = start_pos.clamp(0, ignored_idx)
            end_pos = end_pos.clamp(0, ignored_idx)
                
            ce_loss = nn.CrosEntropyLoss(ignore_index=ignored_idx)
            start_loss = ce_loss(start_logits, start_pos)
            end_loss = ce_loss(end_logits, end_pos)
            
            total_loss = start_loss + end_loss
            
        return total_loss
    
    def _emotion_loss(self, x, emotion_logits):
        loss = None
        labels = x.get('labels', None)
        
        if labels is not None:
            ce_loss = nn.CrosEntropyLoss()
            loss = ce_loss(emotion_logits, labels)
            
        return loss
    

class JointModel(ModelBaseClass):
    def __init__(self, base_model_name: str, no_classes: int):
        """
            base_model_name: str for example SpanBERT/spanbert-base-cased; mrm8488/spanbert-finetuned-squadv2
            no_classes: int 
        """
        
        super(JointModel, self).__init__(base_model_name, no_classes, ModelType.AUTO_MODEL)
        self.hidden_dim_pooler = self.model.pooler.dense.out_features
        self.hidden_dim_last = self.model.encoder.layer[-1].output.dense.out_features
        
        # Span head is linear
        self.span_head = nn.Sequential(
            nn.Linear(self.hidden_dim_last, self.hidden_dim_last // 2),
            nn.GeLU(),
            nn.Linear(self.hidden_dim_last // 2, 2)
        )
        
        # emotion head is non-linear
        self.emotion_head = nn.Sequential(
            nn.Linear(self.hidden_dim_pooler, self.hidden_dim_pooler // 2),
            nn.GeLU(),
            nn.Linear(self.hidden_dim_pooler // 2, no_classes)
        )
        
    def forward(self, x):
        out = self.model(**x)
        hidden_state = out['last_hidden_state']
        pooler_out = out['pooler_output']
        
        span_logits = self.span_head(hidden_state)
        emotion_logits = self.emotion_head(pooler_out)
        
        span_loss = self._span_loss(x, span_logits)
        emotion_loss = self._emotion_loss(x, emotion_logits)
        
        total_loss = None
        
        if span_loss is not None and emotion_loss is not None:
            total_loss = (span_loss + emotion_loss) / 3
        
        return {
            'loss': total_loss,
            'emotion_logits': emotion_logits,
            'span_logits': span_logits
        }
    
    
class EmotionClassifcation(ModelBaseClass):
    def __init__(self, base_model_name: str, no_classes: int):
        """
            base_model_name: str for example "roberta-base"
            no_classes: int 
        """
        
        #TODO:
        # How text is to be fed to the model?
        # - One Scene at a time.
        
        super(EmotionClassifcation, self).__init__(base_model_name, no_classes, ModelType.ROBERTA_SEQUENCE_MODEL)
        
    def forward(self, x):
        
        out = self.model(**x)
        
        return {
            'loss': out.loss,
            'emotion_logits': out.logits
        }
    
class SpanClassification(ModelBaseClass):
    def __init__(self, base_model_name: str, no_classes: int):
        """
            base_model_name: str for example SpanBERT/spanbert-base-cased; mrm8488/spanbert-finetuned-squadv2
            no_classes: int 
        """
        super(SpanClassification, self).__init__(base_model_name, no_classes, ModelType.SPANBERT_MDOEL)
        self.hidden_dim = self.model.pooler.dense.out_features
        self.span_head = nn.Linear(self.hidden_dim, 2)
        
    def forward(self, x):
        out = self.model(**x)
        # We use the hidden state before pooler layers.
        span_logits = self.span_head(out.last_hidden_state)
        loss = self._span_loss(x, span_logits)
        
        if loss is not None:
            loss /= 2
        
        return {
            'loss': loss,
            'span_logits': span_logits
        }
    
