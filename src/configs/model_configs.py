from src.modeling.albef import *
from src.modeling.vilt import *
from src.modeling.vilt_clf import *
from src.modeling.viltbert import *

ALLOWED_CL_ENCODERS = ["vilt", "viltbert", "flava", "albef_distill", "albef_no_distill"]

#### for ViLT
vilt_config = {
    'encoder_dim': 768,
    'visual_input_type': 'pil-image',
    'encoder_class': ViltEncoderWrapper,
    'batch2inputs_converter': convert_batch_to_vilt_input_dict,
    'encoder_name': 'ViLT'
}


viltbert_config = {
    "encoder_dim": 768,
    "visual_input_type": "pil-image",
    "encoder_class": ViltBertEncoderWrapper,
    "batch2inputs_converter": convert_batch_to_viltbert_input_dict,
    "encoder_name": "ViLT-BERT",
}
viltbert_lang_seq_config = {
    "encoder_dim": 768,
    "visual_input_type": "pil-image",
    "encoder_class": ViltBertEncoderWrapper,
    "classifier_class": ViltBertForSequenceClassification,
    "batch2inputs_converter": convert_seq_batch_to_vilt_input_dict,
}
viltbert_lang_mc_config = {
    "encoder_dim": 768,
    "visual_input_type": "pil-image",
    "encoder_class": ViltBertEncoderWrapper,
    "classifier_class": ViltBertForMultipleChoice,
    "batch2inputs_converter": convert_mc_batch_to_vilt_input_dict,
}

config_bert = {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "type_vocab_size": 2,
  "vocab_size": 30522,
  "fusion_layer": 6,
  "encoder_width": 768
}

albef_no_distill_config = {
    "text_encoder": "bert-base-uncased",
    "text_decoder": "bert-base-uncased",
    "image_res": 384,
    "visual_input_type": "pil-image",
    "bert_config": config_bert,
    "batch2inputs_converter": convert_batch_to_albef_input_dict,
    "distill": False,
    "encoder_class": ALBEFWrapper,
    "encoder_name": "albef_no_distill",
}

albef_distill_config = {
    "text_encoder": "bert-base-uncased",
    "text_decoder": "bert-base-uncased",
    "image_res": 384,
    "visual_input_type": "pil-image",
    "bert_config": config_bert,
    "distill": True,
    "batch2inputs_converter": convert_batch_to_albef_input_dict,
    "encoder_class": ALBEFWrapper,
    "encoder_name": "albef_distill",
}

model_configs = {
    "vilt": vilt_config,
    "albef_distill": albef_distill_config,
    "albef_no_distill": albef_no_distill_config,
}
