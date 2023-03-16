import configparser
import os


class ReadConfig:
    """定义一个读取配置文件的类"""

    def __init__(self, filepath=None):
        if filepath:
            configpath = filepath
        else:
            root_dir = os.path.dirname(os.path.abspath('.'))
            configpath = os.path.join(root_dir, "conf.cfg")
        self.cf = configparser.ConfigParser()
        self.cf.read(configpath, encoding="utf-8-sig")

    def getpath(self, param):
        value = self.cf.get("path", param)
        return value


def getpath(key, station):
    root_dir = os.path.dirname(os.path.abspath('.'))
    readcfg = ReadConfig(root_dir + "\\dlbmm\\config\\conf.cfg")
    val = readcfg.getpath('basepath') + station + '\\' + readcfg.getpath('soltype') + '\\'
    if key == 'eachsolpath' or key == 'parsedsolpath' or key == 'parsedpospath' or key == 'parsedstatpath' or \
            key == 'preprocpath' or key == 'cnnworkpath' or key == 'rfrworkpath' or key == 'svrrbfworkpath'  or key == 'cnnviewpath':
        val = val + readcfg.getpath(key)
    if key == 'soltype':
        val = readcfg.getpath(key)
    return val
def loadTemplate(fileName):
    root_dir = os.path.dirname(os.path.abspath('.'))
    filePath = root_dir + "\\dlbmm\\config\\template\\"+fileName
    with open(filePath, 'r', encoding='utf-8') as f:
        content = f.read()
    return content