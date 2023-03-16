import pandas as pd


# load solution  file
def parsesol(solfile):
    data = pd.read_csv(solfile, sep=',', header=None,
                       names=['week', 'tow', 'x', 'y', 'z', 'q', 'ns', 'sdn', 'sde', 'sdu', 'sdene', 'sdeu',
                              'sdun', 'age', 'ratio'], engine='python')
    return data


# parse position from solution status file
def parsepos(solstatusfile):
    datafrm = pd.DataFrame(data=None, index=None,
                           columns=['week', 'tow', 'stat', 'posx', 'posy', 'posz', 'posxf', 'posyf', 'poszf'],
                           dtype=None, copy=False)
    for line in open(solstatusfile):
        if line.startswith('$POS'):
            split_str_eles = line[5:].replace("\n", "").split(',')
            linelement = {'week': int(split_str_eles[0]), 'tow': float(split_str_eles[1]),
                          'stat': int(split_str_eles[2]), 'posx': float(split_str_eles[3]),
                          'posy': float(split_str_eles[4]), 'posz': float(split_str_eles[5]),
                          'posxf': float(split_str_eles[6]), 'posyf': float(split_str_eles[7]),
                          'poszf': float(split_str_eles[8])}
            datafrm = datafrm.append(linelement, ignore_index=True)
    return datafrm


# parse satllite from solution status file
def parsesat(solstatusfile):
    datafrm = pd.DataFrame(data=None, index=None,
                           columns=['week', 'tow', 'sat', 'frq', 'az', 'el', 'resp', 'resc', 'vsat', 'snr', 'fix',
                                    'slip', 'lock', 'outc', 'slipc', 'rejc'],
                           dtype=None, copy=False)
    for line in open(solstatusfile):
        if line.startswith('$SAT'):
            split_str_eles = line[5:].replace("\n", "").split(',')
            linelement = {'week': int(split_str_eles[0]), 'tow': float(split_str_eles[1]), 'sat': split_str_eles[2],
                          'frq': int(split_str_eles[3]), 'az': float(split_str_eles[4]),
                          'el': float(split_str_eles[5]), 'resp': float(split_str_eles[6]),
                          'resc': float(split_str_eles[7]), 'vsat': int(split_str_eles[8]),
                          'snr': float(split_str_eles[9]), 'fix': int(split_str_eles[10]),
                          'slip': int(split_str_eles[11]), 'lock': int(split_str_eles[12]),
                          'outc': int(split_str_eles[13]), 'slipc': int(split_str_eles[14]),
                          'rejc': int(split_str_eles[15])}
            datafrm = datafrm.append(linelement, ignore_index=True)
    return datafrm


def parseposandstatus(solstatusfile):
    posList = []
    satList = []
    for line in open(solstatusfile):
        if line.startswith('$SAT'):
            split_str_eles = line[5:].replace("\n", "").split(',')
            linelement = {'week': int(split_str_eles[0]), 'tow': float(split_str_eles[1]), 'sat': split_str_eles[2],
                          'frq': int(split_str_eles[3]), 'az': float(split_str_eles[4]),
                          'el': float(split_str_eles[5]), 'resp': float(split_str_eles[6]),
                          'resc': float(split_str_eles[7]), 'vsat': int(split_str_eles[8]),
                          'snr': float(split_str_eles[9]), 'fix': int(split_str_eles[10]),
                          'slip': int(split_str_eles[11]), 'lock': int(split_str_eles[12]),
                          'outc': int(split_str_eles[13]), 'slipc': int(split_str_eles[14]),
                          'rejc': int(split_str_eles[15])}
            satList.append(linelement)
        elif line.startswith('$POS'):
            split_str_eles = line[5:].replace("\n", "").split(',')
            linelement = {'week': int(split_str_eles[0]), 'tow': float(split_str_eles[1]),
                          'stat': int(split_str_eles[2]), 'posx': float(split_str_eles[3]),
                          'posy': float(split_str_eles[4]), 'posz': float(split_str_eles[5]),
                          'posxf': float(split_str_eles[6]), 'posyf': float(split_str_eles[7]),
                          'poszf': float(split_str_eles[8])}
            posList.append(linelement)
    posfrm = pd.DataFrame(data=posList, index=None,
                          columns=['week', 'tow', 'stat', 'posx', 'posy', 'posz', 'posxf', 'posyf', 'poszf'],
                          dtype=None, copy=False)
    satfrm = pd.DataFrame(data=satList, index=None,
                          columns=['week', 'tow', 'sat', 'frq', 'az', 'el', 'resp', 'resc', 'vsat', 'snr', 'fix',
                                   'slip', 'lock', 'outc', 'slipc', 'rejc'],
                          dtype=None, copy=False)
    return posfrm, satfrm
