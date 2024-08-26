from easydict import EasyDict as edict

__C = edict()

dataset_cfg = __C

# Breaking Bad geometry assembly dataset

__C.BREAKING_BAD = edict()
__C.BREAKING_BAD.DATA_DIR = "../Breaking-Bad-Dataset.github.io/data/"
__C.BREAKING_BAD.DATA_FN = (  # this would vary due to different subset, use everyday as default
    "everyday.{}.txt"
)
__C.BREAKING_BAD.DATA_KEYS = ("part_ids",)

__C.BREAKING_BAD.SUBSET = ""  # must in ['artifact', 'everyday', 'other']
__C.BREAKING_BAD.CATEGORY = ""  # empty means all categories
__C.BREAKING_BAD.ALL_CATEGORY = [
    "BeerBottle",
    "Bowl",
    "Cup",
    "DrinkingUtensil",
    "Mug",
    "Plate",
    "Spoon",
    "Teacup",
    "ToyFigure",
    "WineBottle",
    "Bottle",
    "Cookie",
    "DrinkBottle",
    "Mirror",
    "PillBottle",
    "Ring",
    "Statue",
    "Teapot",
    "Vase",
    "WineGlass",
]  # Only used for everyday

__C.BREAKING_BAD.ROT_RANGE = -1.0  # rotation range for curriculum learning
__C.BREAKING_BAD.NUM_PC_POINTS = 5000  # points per part
__C.BREAKING_BAD.MIN_PART_POINT = (
    30  # if sampled by area, want to make sure all piece have >30 points
)
__C.BREAKING_BAD.MIN_NUM_PART = 2
__C.BREAKING_BAD.MAX_NUM_PART = 20
__C.BREAKING_BAD.SHUFFLE_PARTS = False
__C.BREAKING_BAD.SAMPLE_BY = "area"

__C.BREAKING_BAD.LENGTH = -1
__C.BREAKING_BAD.TEST_LENGTH = -1
__C.BREAKING_BAD.OVERFIT = -1

__C.BREAKING_BAD.FRACTURE_LABEL_THRESHOLD = 0.025

__C.BREAKING_BAD.COLORS = [
    [0, 204, 0],
    [204, 0, 0],
    [0, 0, 204],
    [127, 127, 0],
    [127, 0, 127],
    [0, 127, 127],
    [76, 153, 0],
    [153, 0, 76],
    [76, 0, 153],
    [153, 76, 0],
    [76, 0, 153],
    [153, 0, 76],
    [204, 51, 127],
    [204, 51, 127],
    [51, 204, 127],
    [51, 127, 204],
    [127, 51, 204],
    [127, 204, 51],
    [76, 76, 178],
    [76, 178, 76],
    [178, 76, 76],
]
