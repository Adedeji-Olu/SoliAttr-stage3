"""Project-wide constants. Anything that changes a result lives here or in a CLI flag."""

N_CLASSES = 11            # gestures 0..10. Gesture 11 (background, no hand) is excluded from the main protocol.
BACKGROUND_GESTURE = 11
N_CHANNELS = 4            # Soli receive channels (ch0..ch3 in the HDF5 files)
RDM_SIZE = 32             # each frame is a 32 x 32 range-Doppler map (stored flat as 1024)
T_FRAMES = 32             # every sequence is resampled to this many frames
GRID = 8                  # pooled grid: 8 time blocks x 8 Doppler bands (each cell = 4 frames x 4 bins)

# Which axis of the reshaped 32x32 map is Doppler. The deep-soli README does not say.
# Check it with `python scripts/inspect_axes.py` before trusting any Doppler-domain perturbation.
DOPPLER_AXIS = 1          # 0 = rows, 1 = columns of the reshaped (32, 32) map

# Session -> user mapping, taken from the deep-soli README ("Dataset session arrangement for evaluation"):
#   cross-user set = sessions 2,3,5,6,8,9,10,11,12,13  (10 users x 25 instances x 11 gestures)
#   single-user cross-session set = sessions 0,1,4,7 (50 each), 13,14 (25 each)
# Session 13 appears in both lists, so the single user ("A") has six sessions and is also the tenth cross-user user.
USER_SESSIONS = {
    "A":   [0, 1, 4, 7, 13, 14],
    "u2":  [2], "u3": [3], "u5": [5], "u6": [6], "u8": [8],
    "u9":  [9], "u10": [10], "u11": [11], "u12": [12],
}
# User-grouped 4-fold protocol. All sessions of a user stay in the same fold.
FOLDS = {0: ["A"], 1: ["u2", "u3", "u5"], 2: ["u6", "u8", "u9"], 3: ["u10", "u11", "u12"]}
N_VAL_USERS = 2           # validation users drawn from the training pool (never user A)
REPEATS = 5               # repeats per fold: different validation users + different model seeds
BUDGET = 4                # hyper-parameter configurations tried per method (same for every method)
