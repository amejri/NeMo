# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import List, Optional

from transformers import (
    ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    ALL_PRETRAINED_CONFIG_ARCHIVE_MAP,
    BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
    AlbertConfig,
    AutoModel,
    BertConfig,
    CamembertConfig,
    DistilBertConfig,
    RobertaConfig,
)

from nemo.collections.nlp.modules.common.huggingface.albert import AlbertEncoder
from nemo.collections.nlp.modules.common.huggingface.bert import BertEncoder
from nemo.collections.nlp.modules.common.huggingface.camembert import CamembertEncoder
from nemo.collections.nlp.modules.common.huggingface.distilbert import DistilBertEncoder
from nemo.collections.nlp.modules.common.huggingface.roberta import RobertaEncoder
from nemo.utils import logging

__all__ = ["get_huggingface_lm_model", "get_huggingface_pretrained_lm_models_list"]


HUGGINGFACE_MODELS = {
    "BertModel": {
        "default": "bert-base-uncased",
        "class": BertEncoder,
        "config": BertConfig,
        "pretrained_model_list": BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    },
    "DistilBertModel": {
        "default": "distilbert-base-uncased",
        "class": DistilBertEncoder,
        "config": DistilBertConfig,
        "pretrained_model_list": DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    },
    "RobertaModel": {
        "default": "roberta-base",
        "class": RobertaEncoder,
        "config": RobertaConfig,
        "pretrained_model_list": ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
    },
    "AlbertModel": {
        "default": "albert-base-v2",
        "class": AlbertEncoder,
        "config": AlbertConfig,
        "pretrained_model_list": ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    },
    "CamembertModel": {
        "default": "camembert-base",
        "class": CamembertEncoder,
        "config": CamembertConfig,
        "pretrained_model_list": CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    }
}


def get_huggingface_lm_model(
    pretrained_model_name: str, config_dict: Optional[dict] = None, config_file: Optional[str] = None,
):
    """
    Returns lm model instantiated with Huggingface

    Args:
        pretrained_mode_name: specify this to instantiate pretrained model from Huggingface,
            e.g. bert-base-cased. For entire list, see get_huggingface_pretrained_lm_models_list().
        config_dict: model configuration dictionary used to instantiate Huggingface model from scratch
        config_file: path to model configuration file used to instantiate Huggingface model from scratch

    Returns:
        BertModule
    """

    try:
        automodel = AutoModel.from_pretrained(pretrained_model_name)
    except Exception as e:
        raise ValueError(f"{pretrained_model_name} is not supported by HuggingFace. {e}")

    model_type = type(automodel).__name__
    if model_type in HUGGINGFACE_MODELS:
        model_class = HUGGINGFACE_MODELS[model_type]["class"]
        if config_file:
            if not os.path.exists(config_file):
                logging.warning(
                    f"Config file was not found at {config_file}. Will attempt to use config_dict or pretrained_model_name."
                )
            else:
                config_class = HUGGINGFACE_MODELS[model_type]["config"]
                return model_class(config_class.from_json_file(config_file))
        if config_dict:
            config_class = HUGGINGFACE_MODELS[model_type]["config"]
            return model_class(config=config_class(**config_dict))
        else:
            return model_class.from_pretrained(pretrained_model_name)
    else:
        raise ValueError(f"Use HuffingFace API directly in NeMo for {pretrained_model_name}")


def get_huggingface_pretrained_lm_models_list(include_external: bool = False,) -> List[str]:
    """
    Returns the list of pretrained HuggingFace language models
    
    Args:
        include_external if true includes all HuggingFace model names, not only those supported language models in NeMo.
    
    Returns the list of HuggingFace models
    """

    huggingface_models = []
    if include_external:
        huggingface_models = list(ALL_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())
    else:
        for model in HUGGINGFACE_MODELS:
            model_names = HUGGINGFACE_MODELS[model]["pretrained_model_list"]
            huggingface_models.extend(model_names)
    return huggingface_models
