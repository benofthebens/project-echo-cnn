import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(ROOT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pt")

MODEL_OUTPUT_DIR = os.path.join(MODEL_DIR, "output")
MODEL_OUPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, "output.wav")

DATA_DIR = os.path.join(ROOT_DIR, "data")

NOISY_TRAIN_DIR = os.path.join(DATA_DIR, "noisy_trainset_wav")
CLEAN_TRAIN_DIR = os.path.join(DATA_DIR, "clean_trainset_wav")

NOISY_TEST_DIR = os.path.join(DATA_DIR, "noisy_testset_wav")
CLEAN_TEST_DIR = os.path.join(DATA_DIR, "clean_testset_wav")




