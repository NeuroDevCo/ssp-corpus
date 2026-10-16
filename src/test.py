import json
from pathlib import Path
from urllib import request

import dendropy
import numpy as np
from phylodm import PhyloDM

tree = dendropy.Tree.get_from_path("data/glottolog.txt", schema="newick")
pdm = PhyloDM.load_from_dendropy(tree)
dm = pdm.dm(norm=False)
labels = pdm.taxa()
