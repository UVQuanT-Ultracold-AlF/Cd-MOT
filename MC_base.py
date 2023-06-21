from init import *
import random as rand
import functools as ft0

gt = lambda x, y : ft.reduce(lambda a, b : a and b, [i >= j for i,j in zip(x,y)])

def rej_samp(func = lambda _ : 1, rand_x = lambda : rand.uniform(0,1), rand_y = lambda : 0, comp_func = lambda x, y : x >= y):
    while True:
        x, y = rand_x(), rand_y()
        # print (x, y, func(x))
        if (comp_func(func(x), y)):
            yield x

def fake_samp(val):
    while True:
        yield val
        
def cdf_samp(cdf, valrange, randrange = None):
    if randrange is None:
        randrange = [cdf(valrange[0]), cdf(valrange[1])]
    while True:
        rand_prop = rand.uniform(*randrange)
        yield bisect(lambda x : cdf(x) - rand_prop,*valrange, xtol = 1e-2/velocity_unit)

abundance_data[107] = 0
abundance_data[109] = 0
abundance_data[115] = 0

# std of 34 corresponds to FWHM of ~80
transverse_pdf = lambda x : norm.pdf(x, 0/velocity_unit, 34/velocity_unit)
transverse_cdf = lambda x : norm.cdf(x, 0/velocity_unit, 34/velocity_unit)
transverse_cutoff = np.tan(35e-3)*200/velocity_unit

mean = 120/velocity_unit
std = 10/velocity_unit

capture_cdf = lambda x : norm.cdf(x, mean, std)
capture_pdf = lambda x : norm.pdf(x, mean, std)