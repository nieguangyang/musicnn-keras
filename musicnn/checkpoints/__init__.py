import os

PATH = "/".join(os.path.realpath(__file__).replace("\\", "/").split("/")[:-1])
MTT_WEIGHTS = PATH + "/mtt_weights.h5"  # MagnaTagATune, https://github.com/keunwoochoi/magnatagatune-list
MSD_WEIGHTS = PATH + "/msd_weights.h5"  # Million Song Dataset, https://github.com/keunwoochoi/MSD_split_for_tagging
