import os

import numpy as np

from analyst.viewhandler import  viewnedpreproc, viewnedpreprocoverview,  viewrfrmodel, \
    viewsvr_rbfmodel, viewerfmodel, analysisrfralldays, analysissvr_rbfalldays, analysiserfalldays, \
    analysiscnn_multialldays, viewallpreproc,viewsvr_rbf_eachmodel

# viewsolemodel('2_day_cnn')
from config.configutil import getpath

# viewnedpreprocoverview(getpath('preproced_path_each'))

# for root, _, wfiles in os.walk(getpath('preproced_path_each')):
#     for file in wfiles:
#         viewnedpreproc(getpath('preproced_path_each'), file)
# viewcnnmultimodel('3_day_cnnmulti')
# 1_1
preproced_path = getpath('preproced_path')
trained_path = getpath("trained_path")
filenames = []
wfiles = os.listdir(preproced_path)
for file in wfiles:
        filenames.append(os.path.join(preproced_path, file))
filenames.reverse()
data = np.load(filenames[0])
# 1_2 view erf
# viewerfmodel(data["feature"], data["label"], "58_erf.pkl")
# 1_2 view svr
viewsvr_rbf_eachmodel(data["feature"], data["label"], "27_svr_rbf.pkl", 1)
viewsvr_rbf_eachmodel(data["feature"], data["label"], "16_svr_rbf.pkl", 2)
viewsvr_rbf_eachmodel(data["feature"], data["label"], "4_svr_rbf.pkl", 3)

# 1_2 view cnn
# viewsolemodel('north' + "45_day_cnn", 1)
# viewsolemodel('east' + "44_day_cnn", 2)
# viewsolemodel('up' + "14_day_cnn", 3)






#viewrfrmodel(data["feature"], data["label"], "3_rfr.pkl")
# viewsvr_rbfmodel(data["feature"], data["label"], "3_svr_rbf.pkl")

# viewcnnmultimodel('3_day_cnnmulti')
# analysisrfralldays(data["feature"], data["label"])
# analysissvr_rbfalldays(data["feature"], data["label"])
# analysiserfalldays(data["feature"], data["label"])
# analysiscnn_multialldays(data["feature"], data["label"])
# analysisrfralldays(data["feature"], data["label"], 'rfr_27')
# analysisrfralldays(data["feature"], data["label"], 'rfr_33')
# analysisrfralldays(data["feature"], data["label"], 'rfr_44')


# viewallpreproc()
# viewsvr_rbf_eachmodel(data["feature"], data["label"], "45_svr_rbf.pkl", 1)
# viewsvr_rbf_eachmodel(data["feature"], data["label"], "47_svr_rbf.pkl", 2)
# viewsvr_rbf_eachmodel(data["feature"], data["label"], "38_svr_rbf.pkl", 3)
# viewerfmodel(data["feature"], data["label"],  "rfr_27.pkl")
# viewerfmodel(data["feature"], data["label"],  "rfr_33.pkl")
# viewerfmodel(data["feature"], data["label"],  "rfr_44.pkl")
# viewsolemodel('north' + "42_day_cnn", 1)
# viewsolemodel('east' + "29_day_cnn", 2)
# viewsolemodel('up' + "43_day_cnn", 3)






