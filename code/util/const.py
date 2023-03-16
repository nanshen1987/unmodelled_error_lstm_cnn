import sys


class _const:
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("Can't change const.%s" % name)

        if not name.isupper():
            raise self.ConstCaseError(
                "const name '%s' is not all uppercase" % name)
        self.__dict__[name] = value

    def __delattr__(self, name):
        if name in self.__dict__:
            raise self.ConstError("can't unbind const(%s)" % name)
        raise NameError(name)
const = _const()
const.WGS_84_SEMI_A = 6378137
const.WGS_84_SEMI_B = 6356752.3142
const.WGS_84_E = 0.0818191908425
