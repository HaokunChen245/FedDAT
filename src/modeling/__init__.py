from .vilt import load_vilt_encoder, create_vilt_continual_learner_model
from .albef import load_albef, create_albef_continual_learner_model
from .viltbert import load_viltbert_encoder, create_viltbert_continual_learner_model

load_encoder_map = {
    'vilt': load_vilt_encoder,
    'viltbert': load_viltbert_encoder,
    'albef_distill': load_albef,
    'albef_no_distill': load_albef
}

create_continual_learner_map = {
    'vilt': create_vilt_continual_learner_model,
    'albef_distill': create_albef_continual_learner_model,
    'albef_no_distill': create_albef_continual_learner_model,
    'viltbert': create_viltbert_continual_learner_model,
}
