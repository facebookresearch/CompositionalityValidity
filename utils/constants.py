import os

BASE_DIR = os.environ.get('BASE_DIR')

MODEL_DIR = os.path.join(BASE_DIR, 'trained_models/')
TMCD_MODEL_DIR = os.path.join(BASE_DIR, 'baseline_replication/TMCD/trained_models/')

DATA_DIR = os.path.join(BASE_DIR, 'data/')
TMCD_DATA_DIR = os.path.join(BASE_DIR, 'baseline_replication/TMCD/data/')

TMCD_DATASETS = {'SCAN', 'geoquery', 'spider'}
dataset_color_mapping = {'COGS': '#313b08', 'SCAN': '#313b08', 'spider': '#520c35', 'geoquery': '#520c35', 'NACS': '#2b3a00'}

MODEL_NICE_NAMES = {
    "lstm_uni": "LSTM Uni",
    "lstm_bi": "LSTM Bi",
    "transformer": "Transformer",
    "t5-base": "T5",
    "bart-base": "BART",
    "btg": "BTG"
}

PRETRAINED_MODEL = {'t5-base', 'bart-base', 'nqg'}
UNPRETRAINED_MODEL = {'lstm_uni', 'lstm_bi', 'transformer', 'btg'} 
SYNTHETIC_DATA = {'SCAN', 'COGS', 'NACS'}
NATURAL_DATA = {'geoquery', 'spider'}

DATASET_NICE_NAMES = {
    "COGS": "COGS",
    "geoquery": "GeoQuery",
    "SCAN": "SCAN",
    "spider": "Spider",
    "NACS": "NACS"
}

SPLIT_NICE_NAMES = {
    "standard": "Std",
    "length": "Length",
    "template": "Template",
    "tmcd": "TMCD",
    "random": "Rand",
    "no_mod-test": "Std-Test",
    "no_mod-gen": "Std-Gen",
    "simple": "Simple",
    "addprim_jump": "Jump",
    "addprim_turn_left": "TurnLeft",
    "template_around_right": "Template",
    "mcd1": "MCD1",
    "mcd2": "MCD2",
    "mcd3": "MCD3",
    "add_jump": "Jump",
    "add_turn_left": "TurnLeft",
    # SCAN Lexical Split Names
    "turn_left_random_cvcv": "TurnLeftRcvcv",
    "turn_left_random_str": "TurnLeftRStr",
    "jump_random_cvcv": "JumpRcvcv",
    "jump_random_str": "JumpRStr",
    # COGS Lexical Split Names
    "no_mod": "Std",
    "random_cvcv-test": "Rcvcv-Test",
    "random_cvcv-gen": "Rcvcv-Gen",
    "random_str-test": "Rstr-Test",
    "random_str-gen": "Rstr-Gen",
    "random_cvcv": "Randcvcv",
    "random_str": "RandStr",
    # GeoQ Lexical Split Names
    "standard_random_cvcv": "Std-Rcvcv",
    "standard_random_str": "Std-Rstr",
    "tmcd_random_cvcv": "TMCD-Rcvcv",
    "tmcd_random_str": "TMCD-Rstr",
}

default_model_names = ['lstm_uni', 'lstm_bi', 'transformer', 't5-base', 'bart-base', 'btg']
default_dataset_mapping = { 
    "NACS": ["simple", "add_jump", "add_turn_left", "length"],
                          "spider": ["random", "template", "tmcd", "length"],
                    "COGS": ["no_mod-test", "no_mod-gen"],
                    "geoquery": ["standard", "template", "tmcd", "length"], 
                     
                    "SCAN": ["simple", "addprim_jump", "addprim_turn_left", "template_around_right", "mcd1", "mcd2", "mcd3", "length"], 
                    
                    }

all_dataset_mapping = {"COGS": ["no_mod-test", "random_cvcv-test", "random_str-test",
                                 "no_mod-gen", "random_cvcv-gen", "random_str-gen", "length"], 
                    "spider": ["random", "template", "tmcd", "length"],
                    "SCAN": ["simple", "addprim_jump", "template_around_right", "mcd1", "mcd2", "mcd3", "length", "addprim_turn_left", "turn_left_random_cvcv", "turn_left_random_str"], 
                    "geoquery": ["standard", "standard_random_cvcv", "standard_random_str", "template", "length", "tmcd", "tmcd_random_cvcv", "tmcd_random_str"],
                    "NACS": ["simple", "add_jump", "add_turn_left", "length"],
                    }
lexical_dataset_mapping = {
    "COGS": ["no_mod-gen", "random_cvcv-test", "random_cvcv-gen", "random_str-test", "random_str-gen"],
    "SCAN": ["addprim_turn_left", "turn_left_random_cvcv", "turn_left_random_str"],
    "geoquery": ["standard", "standard_random_cvcv", "standard_random_str", "tmcd", "tmcd_random_cvcv", "tmcd_random_str"]
}

length_dataset_mapping = {
    "COGS": ["length"],
    "SCAN": ["length"],
    "geoquery": ["length"],
    "spider": ["length"],
    "NACS": ["length"]
}

all_exclude_length_dataset_mapping = {"COGS": ["no_mod-test", "random_cvcv-test", "random_str-test",
                                 "no_mod-gen", "random_cvcv-gen", "random_str-gen"], 
                    "spider": ["random", "template", "tmcd", "length"],
                    "SCAN": ["simple", "length", "addprim_jump", "template_around_right", "mcd1", "mcd2", "mcd3", "addprim_turn_left", "turn_left_random_cvcv", "turn_left_random_str"], 
                    "geoquery": ["standard", "standard_random_cvcv", "standard_random_str", "template", "length", "tmcd", "tmcd_random_cvcv", "tmcd_random_str"],
                    "NACS": ["simple", "add_jump", "add_turn_left", "length"],
                    }

raw_dataset_mapping = {
                    "COGS": ["no_mod-test", 
                                 "no_mod-gen"], 
                    "spider": ["random", "template", "tmcd", "length"],
                    "SCAN": ["simple", "addprim_jump", "template_around_right", "mcd1", "mcd2", "mcd3", "length", "addprim_turn_left"], 
                    "geoquery": ["template", "length", "standard", "tmcd"],
                    "NACS": ["simple", "add_jump", "add_turn_left", "length"],
                    }
lexical_without_orig_dataset_mapping = {
                    "COGS": ["random_cvcv-test", "random_str-test", "random_cvcv-gen", "random_str-gen"], 
                    "spider": ["random", "template", "tmcd", "length"],
                    "SCAN": ["simple", "addprim_jump", "template_around_right", "mcd1", "mcd2", "mcd3", "length", "addprim_turn_left", "turn_left_random_cvcv"], 
                    "geoquery": ["template", "length",  "standard_random_cvcv", "standard_random_str", "tmcd_random_cvcv", "tmcd_random_str"],
                    "NACS": ["simple", "add_jump", "add_turn_left", "length"],
                    }
lexical_w_orig_mapping = {
    "COGS": {
        # "no_mod-test": ["random_cvcv-test", "random_str-test"],
            "no_mod-gen": ["random_cvcv-gen", "random_str-gen"]}, 
    "SCAN": {"addprim_turn_left":  ["addprim_turn_left", "turn_left_random_cvcv"]}, 
    "geoquery": {
        # "standard": ["standard_random_cvcv", "standard_random_str"], 
        "tmcd": ["tmcd_random_cvcv", "tmcd_random_str"]},
}