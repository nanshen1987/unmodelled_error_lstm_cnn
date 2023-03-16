from math import sqrt, cos, sin, atan, fabs, pi
import numpy as np


class CoordinateTransform:
    """coordinate transform"""

    def __init__(self, semi_a, semi_b):
        self.semi_a = semi_a
        self.semi_b = semi_b
        self.eccentricity = sqrt(semi_a * semi_a - semi_b * semi_b)

    def blh2xyz(self, blh):
        B = blh[0]
        L = blh[1]
        H = blh[2]
        N = self.semi_a * self.semi_a / sqrt(
            self.semi_a * self.semi_a * cos(B) * cos(B) + self.semi_b * self.semi_b * sin(B) * sin(B));
        X = (N + H) * cos(B) * cos(L);
        Y = (N + H) * cos(B) * sin(L);
        Z = (self.semi_b * self.semi_b * N / (self.semi_a * self.semi_a) + H) * sin(B);
        return np.array([X, Y, Z])

    def xyz2blh(self, xyz):
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]
        x2 = x*x
        y2 = y*y
        z2 = z*z

        a = self.semi_a
        b = self.semi_b
        e = sqrt(1 - (b / a) *(b / a))
        b2 = b * b
        e2 = e *e
        ep = e * (a / b)
        r = sqrt(x2 + y2)
        r2 = r * r
        E2 = a*a - b*b
        F = 54 * b2 * z2
        G = r2 + (1 - e2) * z2 - e2 * E2
        c = (e2 * e2 * F * r2) / (G * G * G)
        s = np.power((1 + c + sqrt(c * c + 2 * c)),(1.0 / 3))
        P = F / (3 * np.power(s + 1 / s + 1, 2) * G * G)
        Q = sqrt(1 + 2 * e2 * e2 * P)
        ro = -(P * e2 * r) / (1 + Q) + sqrt((a * a / 2) * (1 + 1 / Q)- (P * (1 - e2) * z2) / (Q * (1 + Q)) - P * r2 / 2)
        tmp = (r - e2 * ro) *(r - e2 * ro)
        U = sqrt(tmp + z2)
        V = sqrt(tmp + (1 - e2) * z2)
        zo = (b2 * z) / (a * V)

        height = U * (1 - b2 / (a * V))

        lat = atan((z + ep * ep * zo) / r)

        temp = atan(y / x)
        if x >= 0:
            long = temp
        elif(x < 0) & (y >= 0):
            long = pi + temp
        else:
            long = temp - pi

        return np.array([lat, long, height])

    def xyz2ned(self, xyz, bl=None):
        if bl.any():
            B = bl[0]
            L = bl[1]
        else:
            (B, L, H) = self.xyz2blh(xyz)
        if L < 0:
            L += pi
        sL = sin(L)
        cL = cos(L)
        sB = sin(B)
        cB = cos(B)
        rot = np.mat(np.zeros((3, 3)))
        rot[0, 0] = -sB * cL
        rot[0, 1] = -sB * sL
        rot[0, 2] = cB
        rot[1, 0] = -sL
        rot[1, 1] = cL
        rot[1, 2] = 0
        rot[2, 0] = -cB * cL
        rot[2, 1] = -cB * sL
        rot[2, 2] = -sB
        return np.matmul(rot, xyz)

    def blh2ned(self, blh, bl=None):
        xyz = self.blh2xyz(blh)
        return self.xyz2ned(xyz, bl)
