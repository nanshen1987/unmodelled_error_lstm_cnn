import os
from rtklib.solutils import parseposandstatus, parsesol
"""解析rtklib定位文件和状态文件"""
def pasredsol(eachsolpath, parsedsolpath, parsedpospath, parsedstatpath):
    solstatusfileExt = ".sol.stat"
    solext = ".sol"
    for root, _, files in os.walk(eachsolpath):
        for file in files:
            filename = file[0:file.find('.')]
            if file.endswith(solext):
                solfrm = parsesol(eachsolpath + file)
                solfrm.to_pickle(parsedsolpath + filename + "_sol" + ".pkl")
            elif file.endswith(solstatusfileExt):
                (posfrm, satfrm) = parseposandstatus(eachsolpath + file)
                posfrm.to_pickle(parsedpospath + filename + "_pos" + ".pkl")
                satfrm.to_pickle(parsedstatpath + filename + "_sat" + ".pkl")