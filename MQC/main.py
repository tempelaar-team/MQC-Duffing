import os
from shutil import copyfile
from input import *

def main():
    if status=='RESTART':
        global calcdir
        calcdir = calcdir + '/RESTART'
        print('RESTART with endpoints of Q, P, and wavefn')
        os.mkdir(calcdir)
    else:
        if not (os.path.exists(calcdir)):
            os.mkdir(calcdir)
    copyfile('input.py', calcdir + '/inputfile_bk.txt')
    from mixQC_MF_Foc import runCalc
    runCalc()

if __name__ == '__main__':
    main()