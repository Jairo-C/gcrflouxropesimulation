import math
import numpy as np
from T1.constantsModule import d0p, d1p, d2p, dp5, eV2J, m2AU, AU2m, deg2rad,c, d180p
from scipy.special import jn as bessel_jn

def vele(energia, masa):
    """
    En esta función se calcula la velocidad de una partícula con una energía dada.
    La deducción es explicada en el documento de la tesis, básicamente se despeja 
    la velocidad de la ecuación de energía: E = gama * m * c^2, donde gama es el factor de Lorentz
    """
    if energia <= 0:
        raise ValueError("La energía debe ser un valor positivo.")
        
    argumento = 1 - ((masa * c**2) / energia)**2

    if argumento < 0:
        raise ValueError("Se está obteniendo un término negativo para evaluar la raíz.")
    
    velocidade = c * math.sqrt(argumento)

    return velocidade

def factor_de_lorentz(v):
    """
    En esta función se calcula el factor de Lorentz de una partícula con una velocidad dada.
    """
    if v >= c:
        raise ValueError("La velocidad de la partícula no puede ser más grande que c.")
    
    return 1 / math.sqrt(1 - (v / c)**2)

def cross(a, b):
    """
    Calculamos con esta función el producto cruz de dos vectores de 3 dimensiones.
    """
    a = np.array(a)
    b = np.array(b)
    
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("Los dos vectores deben ser de 3 dimensiones.")
    
    productocruz = np.array([
        a[1] * b[2] - a[2] * b[1],
        -a[0] * b[2] + a[2] * b[0],
        a[0] * b[1] - a[1] * b[0]
    ])
    
    return productocruz

def stduniformedistr():
    """
    Se genera un número aleatorio de una distribución uniforme estándar.    
    """
    r = np.random.random()
    return 1.0 - r

def stdnormaldistr():
    """
    Se genera un número aleatorio de una distribución normal estándar.
    """
    u1 = stduniformedistr()
    u2 = stduniformedistr()
    x = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    return x

def stdnormalmeandistr(mean, std):
    """
    Se genera un número aleatorio de una distribución normal con media y desviación estándar dadas.
    """
    x = stdnormaldistr()
    return std * x + mean


def magnetic_field(x, y, z, pp, BFld, err):
    """
    Calculate the magnetic field at a given point.

    Parameters:
    x (float): x-coordinate.
    y (float): y-coordinate.
    z (float): z-coordinate.
    pp (PusherParams): Instance of PusherParams containing the parameters.

    Returns:
    tuple: A tuple containing the magnetic field vector (BFld) and an error code (err).
    """
    BFld = np.zeros(3)
    err = 0
    cosTita = x / np.sqrt(x**2 + y**2)

    if pp.addBFldNoise:
        if pp.fieldShape.strip() == 'UNIFORM':
            tempB = stdnormalmeandistr(mean=pp.BswMean, std=pp.BswNoiseStd)
            BFld[0] = pp.dirBsw[0] * tempB
            BFld[1] = pp.dirBsw[1] * tempB
            BFld[2] = pp.dirBsw[2] * tempB
        elif pp.fieldShape.strip() == 'BESSEL':
            if x**2 + y**2 <= pp.frRadius**2:
                alfa = pp.alfa
                H = pp.H
                rho = np.sqrt(x**2 + y**2)
                Bo = pp.BfrMean
                noise = np.array([stdnormalmeandistr(mean=0.0, std=pp.BfrNoiseStd) for _ in range(3)])
                Bphi = Bo * H * bessel_jn(1, alfa * rho)
                phi = np.arctan2(y, x)
                BFld[0] = -Bphi * np.sin(phi) + noise[0]
                BFld[1] = Bphi * np.cos(phi) + noise[1]
                BFld[2] = Bo * bessel_jn(0, alfa * rho) + noise[2]
            else:
                noise = np.array([stdnormalmeandistr(mean=0.0, std=pp.BswNoiseStd) for _ in range(3)])
                BFld[0] = pp.dirBsw[0] * pp.BswMean + noise[0]
                BFld[1] = pp.dirBsw[1] * pp.BswMean + noise[1]
                BFld[2] = pp.dirBsw[2] * pp.BswMean + noise[2]
        elif pp.fieldShape.strip() == 'BESSEL_SHEATH':
            xparable = parableSimX(x0=pp.focalDist, y0=0.0, a=-pp.focalDist, y=y)
            if x**2 + y**2 <= pp.frRadius**2:
                alfa = pp.alfa
                H = pp.H
                rho = np.sqrt(x**2 + y**2)
                Bo = pp.BfrMean
                noise = np.array([stdnormalmeandistr(mean=0.0, std=pp.BfrNoiseStd) for _ in range(3)])
                Bphi = Bo * H * bessel_jn(1, alfa * rho)
                phi = np.arctan2(y, x)
                BFld[0] = -Bphi * np.sin(phi) + noise[0]
                BFld[1] = Bphi * np.cos(phi) + noise[1]
                BFld[2] = Bo * bessel_jn(0, alfa * rho) + noise[2]
            elif x > xparable:
                noise = np.array([stdnormalmeandistr(mean=0.0, std=pp.BswNoiseStd) for _ in range(3)])
                BFld[0] = pp.dirBsw[0] * pp.BswMean + noise[0]
                BFld[1] = pp.dirBsw[1] * pp.BswMean + noise[1]
                BFld[2] = pp.dirBsw[2] * pp.BswMean + noise[2]
            elif 0.0 <= x <= xparable:
                noise = np.array([stdnormalmeandistr(mean=0.0, std=pp.BswNoiseStd) for _ in range(3)])
                Bsw = pp.dirBsw * pp.BswMean + noise
                noise = np.array([stdnormalmeandistr(mean=0.0, std=pp.BshNoiseStd) for _ in range(3)])
                BFld = (pp.dirBsw * (pp.BshMean - pp.BswMean) + noise) * cosTita + Bsw
            else:
                noise = np.array([stdnormalmeandistr(mean=0.0, std=pp.BswNoiseStd) for _ in range(3)])
                BFld[0] = pp.dirBsw[0] * pp.BswMean + noise[0]
                BFld[1] = pp.dirBsw[1] * pp.BswMean + noise[1]
                BFld[2] = pp.dirBsw[2] * pp.BswMean + noise[2]
        else:
            write_log("ERROR. Magnetic field shape not specified.", 0)
            err += 1
    else:
        if pp.fieldShape.strip() == 'UNIFORM':
            tempB = pp.BswMean
            BFld[0] = pp.dirBsw[0] * tempB
            BFld[1] = pp.dirBsw[1] * tempB
            BFld[2] = pp.dirBsw[2] * tempB
        elif pp.fieldShape.strip() == 'BESSEL':
            if x**2 + y**2 <= pp.frRadius**2:
                alfa = pp.alfa
                H = pp.H
                rho = np.sqrt(x**2 + y**2)
                Bo = pp.BfrMean
                Bphi = Bo * H * bessel_jn(1, alfa * rho)
                phi = np.arctan2(y, x)
                BFld[0] = -Bphi * np.sin(phi)
                BFld[1] = Bphi * np.cos(phi)
                BFld[2] = Bo * bessel_jn(0, alfa * rho)
            else:
                tempB = pp.BswMean
                BFld[0] = pp.dirBsw[0] * tempB
                BFld[1] = pp.dirBsw[1] * tempB
                BFld[2] = pp.dirBsw[2] * tempB
        elif pp.fieldShape.strip() == 'BESSEL_SHEATH':
            xparable = parableSimX(x0=pp.focalDist, y0=0.0, a=-pp.focalDist, y=y)
            if x**2 + y**2 <= pp.frRadius**2:
                alfa = pp.alfa
                H = pp.H
                rho = np.sqrt(x**2 + y**2)
                Bo = pp.BfrMean
                Bphi = Bo * H * bessel_jn(1, alfa * rho)
                phi = np.arctan2(y, x)
                BFld[0] = -Bphi * np.sin(phi)
                BFld[1] = Bphi * np.cos(phi)
                BFld[2] = Bo * bessel_jn(0, alfa * rho)
            elif x > xparable:
                tempB = pp.BswMean
                BFld[0] = pp.dirBsw[0] * tempB
                BFld[1] = pp.dirBsw[1] * tempB
                BFld[2] = pp.dirBsw[2] * tempB
            elif 0.0 <= x <= xparable:
                tempB = pp.BswMean
                Bsw = pp.dirBsw * tempB
                tempB = pp.BshMean - pp.BswMean
                BFld = pp.dirBsw * tempB * cosTita + Bsw
            else:
                tempB = pp.BswMean
                BFld[0] = pp.dirBsw[0] * tempB
                BFld[1] = pp.dirBsw[1] * tempB
                BFld[2] = pp.dirBsw[2] * tempB
        else:
            write_log("ERROR. Magnetic field shape not specified.", 0)
            err += 1

    return BFld, err
