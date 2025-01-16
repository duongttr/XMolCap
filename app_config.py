import os

LPM_24_CFG_PATH = os.path.join(
    os.path.dirname(__file__), 'src/configs/config_lpm24_train.yaml'
)
LPM_24_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), 'weights/lpm24.ckpt'
)
CHEBI_20_CFG_PATH = os.path.join(
    os.path.dirname(__file__), 'src/configs/config_chebi20_train.yaml'
)
CHEBI_20_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), 'weights/chebi20.ckpt'
)