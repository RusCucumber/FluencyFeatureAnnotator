from pathlib import Path

from modules.common.clause import Clause  # noqa: F401
from modules.common.gcp import GCSBucket  # noqa: F401
from modules.common.turn import Turn, shorten_turn  # noqa: F401
from modules.common.word import DisfluencyEnum, DisfluencyWord, Word  # noqa: F401

RESOURCES_PATH = Path.cwd() / "src/fluencyfeatureannotator/modules/resources"
if not RESOURCES_PATH.exists():
    RESOURCES_PATH = Path(__file__).parents[2] / "modules/resources"

