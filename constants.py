# Folders

DATA_FOLDER = "./data"
RAW_DATA_FOLDER = f"{DATA_FOLDER}/raw"
DRUGCOMB_DATA_FOLDER = f"{DATA_FOLDER}/drugcomb"
ONEIL_DATA_FOLDER = f"{DATA_FOLDER}/oneil"
FOLDS_FOLDER_NAME = f"folds"
TEST_RESULT_FOLDER = f"{DATA_FOLDER}/test_results"
CHECKPOINTS_FOLDER = "./checkpoints"

# Files

DRUGCOMB_FILE_NAME = "drugcomb_loewe.feather"
CELL_LINE_FILE_NAME = "cell_lines.feather"
RAW_CELL_LINE_FILE_NAME = "Cell_line_RMA_proc_basalExp.txt"
COSMIC_IDS_FILE_NAME = "cellosaurus_cosmic_ids.txt"
RAW_DRUGCOMB_FILE_NAME = "summary_v_1_5.csv"
DRUGS_FILE_NAME = "drugs.json"

# Configs

TRANSFORMER_BASELINE_CONFIG = "./config/transformer_config.json"
GRAPH_BASELINE_CONFIG = "./config/graph_config.json"

# DataFrame Column Names

DRUG1_COLUMN_NAME = "Drug1"
DRUG2_COLUMN_NAME = "Drug2"
CELL_LINE_COLUMN_NAME = "Cell_Line_ID"
TARGET_COLUMN_NAME = "target"

DDI_TARGET_COLUMN_NAME = "Y"

# Keys

DRUG1_KEY = "Drug1"
DRUG2_KEY = "Drug2"
DRUG1_ID_KEY = "Drug1_ID"
DRUG2_ID_KEY = "Drug2_ID"

CELL_LINE_KEY = "CELL_LINE"
TARGET_KEY = "TARGET"
OUTPUT_KEY = "OUTPUT"

# Checkpoints

MOLCLR_CKPT = f"{CHECKPOINTS_FOLDER}/MolCLR/model.pth"

# Encoder

DRUG_ENCODER = "DeepChem/ChemBERTa-10M-MLM"
ENCODER_OUTPUT_DIM = 384
CELL_LINE_DIM = 908

# Utils

RANDOM_SEED = 42

SYNERGY_SCORES = ["loewe", "bliss", "hsa", "zip"]
