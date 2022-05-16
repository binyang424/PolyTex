import pandas as pd
import os, shutil
import numpy as np
import matplotlib.pyplot as plt


def buildKrigFunc(x,xKnown,B,deriveFuncs,covFuncs):
    funcKrig=0
    for i in range (len(xKnown)):
        funcKrig=covFuncs(h(x,xKnown[i]))*B[i]+funcKrig
    for i in range (len(B)-len(xKnown)):
        funcKrig=funcKrig+B[i+len(xKnown)]*deriveFuncs(x)[i]
    return funcKrig


def buildKrigFunc_deriv(x,xKnown,xKnown_deriv,B,deriveFuncs,covFuncs,covFuncs_deriv):
    funcKrig=0
    for i in range (len(xKnown)):
        funcKrig=covFuncs(h(x,xKnown[i]))*B[i]+funcKrig
        
    for i in range (len(xKnown),len(xKnown)+len(xKnown_deriv)):
        funcKrig=covFuncs_deriv(h(x,xKnown_deriv[i-len(xKnown)]))*B[i]*fn_sign(xKnown_deriv[i-len(xKnown)]-x)+funcKrig
        
    for i in range (len(xKnown)+len(xKnown_deriv),len(B)):
        funcKrig=funcKrig+B[i]*deriveFuncs(x)[i-(len(xKnown)+len(xKnown_deriv))]
    return funcKrig


def build_D_KrigFunc(x,xKnown,B,deriveFuncs,covFuncs):
    D_funcKrig=0
    for i in range (len(xKnown)):
        xx = symbols('xx')
        if x>xKnown[i]:
            D_funcKrig=diff(covFuncs(xx-xKnown[i])*B[i],xx).evalf(subs ={'xx':x})+D_funcKrig
        else:
            D_funcKrig=diff(covFuncs(-xx+xKnown[i])*B[i],xx).evalf(subs ={'xx':x})+D_funcKrig
    for i in range (len(B)-len(xKnown)):
        xx = symbols('xx')
        D_funcKrig=D_funcKrig+diff(B[i+len(xKnown)]*deriveFuncs(xx)[i],xx).evalf(subs ={'xx':x})
    return D_funcKrig


def resetNoArray(array_tmp):
    array_aft=np.zeros(len(array_tmp))
    array_aft[:len(array_tmp)]=array_tmp
    return array_aft


def buildU(y,deriveFuncs):
    ylen, deriveFuncLen=len(y), len(deriveFuncs(y[0]))
    lenMatU=deriveFuncLen+ylen
    U=np.zeros(lenMatU)
    U[:ylen]=y
    print('Matrix u writes:')
    print(U)
    return U


def buildU_deriv(y,y_deriv,deriveFuncs):
    ylen, ylen_deriv, deriveFuncLen=len(y), len(y_deriv), len(deriveFuncs(y[0]))
    lenMatU=deriveFuncLen+ylen+ylen_deriv
    U=np.zeros(lenMatU)
    U[:ylen]=y
    U[ylen:ylen+ylen_deriv]=y_deriv
    print('Matrix u writes:')
    print(U)
    return U


def solveB(M,U):
    B=np.linalg.solve(M,U)
    print('solution Matrix b writes:')
    print(B)
    return B


def h(x1,x2):
	return np.abs(x1-x2)

    
def fn_sign(x):
    if x<0 :
        res=-1
    else:
        res=1
    return res


def h_2d(x1,x2,y1,y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5
def open_file(file_loc, index_col_nu, delim_whitespace_tf):
    data = pd.read_csv(file_loc,names=[i for i in range(index_col_nu)], delim_whitespace=delim_whitespace_tf)
    return data


def buildM(x,deriveFuncs,covFuncs,nugg):
    xlen, deriveFuncLen=len(x), len(deriveFuncs(x[0]))
    lenMatM=deriveFuncLen+xlen
    M=np.zeros((lenMatM,lenMatM))
    for i in range(xlen):
        for j in range(xlen):
            if i==j:
                M[i,j]=0+nugg
            else:
                M[i,j]=covFuncs(h(x[i],x[j]))
    for i in range(xlen):
        for j in range(deriveFuncLen):
            M[i,j+xlen]=deriveFuncs(x[i])[j]
            M[j+xlen,i]=deriveFuncs(x[i])[j]
    print('Solve M*b=u to obtain Kriging function')
    print('Matrix M writes:')
    print(M)		
    return M


def buildM_deriv(x,x_deriv,deriveFuncs,covFuncs,covFuncs_deriv,covFuncs_deriv2,nugg):
    xlen, xlen_deriv, deriveFuncLen=len(x), len(x_deriv), len(deriveFuncs(x[0]))
    lenMatM=deriveFuncLen+xlen+xlen_deriv
    M=np.zeros((lenMatM,lenMatM))
    for i in range(xlen):
        for j in range(xlen):
            if i==j:
                M[i,j]=0+nugg
            else:
                M[i,j]=covFuncs(h(x[i],x[j]))
    for i in range(xlen):
        for j in range(xlen_deriv):
            M[i,j+xlen]=covFuncs_deriv(x[i]-x_deriv[j])*fn_sign(-x[i]+x_deriv[j])
            M[j+xlen,i]=covFuncs_deriv(x[i]-x_deriv[j])*fn_sign(-x[i]+x_deriv[j])
    for i in range(xlen_deriv):
        for j in range(xlen_deriv):
            M[i+xlen,j+xlen]=covFuncs_deriv2(x_deriv[i]-x_deriv[j])*fn_sign(-x_deriv[i]+x_deriv[j])
            M[j+xlen,i+xlen]=covFuncs_deriv2(x_deriv[i]-x_deriv[j])*fn_sign(-x_deriv[i]+x_deriv[j])
    for i in range(xlen):
        for j in range(deriveFuncLen):
            M[i,j+xlen+xlen_deriv]=deriveFuncs(x[i])[j]
            M[j+xlen+xlen_deriv,i]=deriveFuncs(x[i])[j]
    print('Solve M*b=u to obtain Kriging function')
    print('Matrix M writes:')
    print(M)		
    return M


def open_file2(file_loc,index_col_nu):
    data = pd.read_csv(file_loc,names=[i for i in range(index_col_nu)])
#    data=np.array(data)
#    data=data[:,0]
    return data


def creat_dataframe_vide(col,ind):
    elsets = pd.DataFrame(columns=[i for i in range(col)],index=[i for i in range(ind)])
    #elsets=elsets.astype(str)
    return elsets


def CaseFolderCreate(CaseFolderName,isDelete):
    isExists=os.path.exists(CaseFolderName)
    if isExists and isDelete:
        shutil.rmtree(CaseFolderName)
    os.makedirs(CaseFolderName)
    os.makedirs(CaseFolderName+'/0')
    os.makedirs(CaseFolderName+'/constant')
    os.makedirs(CaseFolderName+'/system')
    open(CaseFolderName+'/foam.foam','w')

    
def writeExe(fileloc):
    fileloc=fileloc+'/exeCase'
    insert_pd=creat_dataframe_vide(1,2)
    insert_pd.iloc[0,0]='blockMesh'
    insert_pd.iloc[1,0]='darcy2TempFoam'
    insert_pd.to_csv(fileloc, header=False, index=False)
    os.system('chmod+x '+fileloc)


def writeExeRunAll(fileloc,n):
    n=n+1
    fileloc='exeCaseAll'
    insert_pd=creat_dataframe_vide(1,3*n-3)
    for i in range (1,n):
        insert_pd.iloc[3*(i-1),0]='cd Case'+str(i)+'/'
        insert_pd.iloc[3*(i-1)+1,0]='./exeCase'
        insert_pd.iloc[3*(i-1)+2,0]='cd ..'
    insert_pd.to_csv(fileloc, header=False, index=False)
    os.system('chmod+x '+fileloc)
    print('./exeCaseAll')
    os.system('xfce4-terminal')


def runPostPro(nCases,npt,Varb,TempRange):
    n=nCases+1
    q=0
    Cr_time=creat_dataframe_vide(2,nCases)
    for i in range (1,n):
        writedown=True
        fileloc='Case'+str(i)+'/postProcessing/probes/0/'+Varb
        file_post=open_file(fileloc,npt,'double',npt+2,0,None)
        file_post_min=file_post.min(axis=1)        
        file_post=open_file(fileloc,npt,'double',npt+2,0,False)
        file_post_max=file_post.max(axis=1)
        totLine=file_post.shape[0]
        insert_pd=creat_dataframe_vide(3,totLine)
        for j in range(totLine):
            insert_pd.iloc[j,0]=file_post.iloc[j,0]
            insert_pd.iloc[j,1]=file_post_max.iloc[j]
            insert_pd.iloc[j,2]=file_post_min.iloc[j]
        for j in range(totLine):
            if abs(insert_pd.iloc[j,1]-insert_pd.iloc[j,2])<TempRange and j>10 and writedown:
                Cr_time.iloc[q,0]='Case'+str(i)
                Cr_time.iloc[q,1]=insert_pd.iloc[j,0]
                q=q+1
                writedown=False
        insert_pd.to_csv(fileloc+'_MaxMin', header=False, index=False, sep=' ')
        Cr_time.to_csv('Cases_MaxMin_CrTime.txt', header=False, index=False, sep=' ')
    return file_post, Cr_time


def bd_kriging_func(x,x_var_sym,y,choixDerive,choixCov,kringFunctionStr,plot_x_pts,nugg):
    plt.scatter(x,y,color='k',marker='x',alpha=0.7,s=12,label='Tf')
    #---------------Choix de la dérive-----------------------
    deriveFuncs={'cst': lambda x: [1], 'lin': lambda x: [1,x]}#, 'quad': lambda x: [1,x,x**2.]}
    #-------------------Choix de la covariance---------------
    covFuncs={'lin': lambda x: x, 'cub': lambda x: x**3.}
    #---------------------Build matrix M---------------------
    M=buildM(x, deriveFuncs[choixDerive], covFuncs[choixCov],nugg)
    #---------------------Build matrix u--------------
    U1=buildU(y, deriveFuncs[choixDerive])
    #----------------------solve b--------------------
    B1=solveB(M,U1)
    #---------------------build string function in c--------------------
    covStr=''
    if choixDerive=='lin':
        drfiNo=2
        driftStr=str(B1[-drfiNo])+'+'+str(B1[-drfiNo+1])+'*'+x_var_sym
    if choixDerive=='cst':
        drfiNo=1
        driftStr=str(B1[-drfiNo])
    if choixDerive=='quad':
        drfiNo=3
        driftStr=str(B1[-drfiNo])+'+'+str(B1[-drfiNo+1])+'*'+x_var_sym+'+'+str(B1[-drfiNo+2])+'*'+x_var_sym+'*'+x_var_sym
    #--
    if choixCov=='lin':
        for i in range (len(x)):
            covStr=covStr+str(B1[i])+'*'+'fabs('+x_var_sym+'-'+str(x[i])+')'+'+'
    if choixCov=='cub':
        for i in range (len(x)):
            covStr=covStr+str(B1[i])+'*'+'pow('+'fabs('+x_var_sym+'-'+str(x[i])+')'+',3.)'+'+'
    kringFunctionStr=covStr+driftStr
    lowerX, upperX=min(x), max(x)
    intervalX=(upperX-lowerX)/plot_x_pts
    x_krig=[i*intervalX for i in range (int(lowerX/intervalX),int((upperX)/intervalX)+1)]
    y_krig=[buildKrigFunc(x_krig[i],x,B1,deriveFuncs[choixDerive], covFuncs[choixCov]) for i in range (len(x_krig))]
    plt.plot(x_krig,y_krig,linestyle='--',lw=1,label='Nugget effect = '+str(nugg))
    return kringFunctionStr, x_var_sym


def bd_Deriv_kriging_func(x,y,x_deriv,y_deriv,choixDerive,choixCov,plot_x_pts,nugg):
    plt.scatter(x,y,color='k',marker='x',alpha=0.7,s=12,label='')
    #---------------Choix de la dérive-----------------------
    deriveFuncs={'cst': lambda x: [1], 'lin': lambda x: [1,x]}
    #-------------------Choix de la covariance---------------
    covFuncs={'cub': lambda x: x**3.,'lin': lambda x: x}
    covFuncs_deriv={'cub': lambda x: 3*x**2.,'lin': lambda x: x**0}
    covFuncs_deriv2={'cub': lambda x: 6*x**1.,'lin': lambda x: x*0}
    #---------------------Build matrix M---------------------
    M=buildM_deriv(x,x_deriv, deriveFuncs[choixDerive], covFuncs[choixCov], covFuncs_deriv[choixCov],covFuncs_deriv2[choixCov],nugg)
    #---------------------Build matrix u--------------
    U1=buildU_deriv(y,y_deriv, deriveFuncs[choixDerive])
    #----------------------solve b--------------------
    B1=solveB(M,U1)
    #---------------------build string function in c--------------------
    lowerX, upperX=min(x), max(x)
    intervalX=(upperX-lowerX)/plot_x_pts
    x_krig=[i*intervalX for i in range (int(lowerX/intervalX),int((upperX)/intervalX)+1)]
    y_krig=[buildKrigFunc_deriv(x_krig[i],x,x_deriv,B1,deriveFuncs[choixDerive], covFuncs[choixCov],covFuncs_deriv[choixCov]) for i in range (len(x_krig))]
    sum_ave=0
    for i in range(1,len(x_krig)):
        hh=x_krig[i]-x_krig[i-1]
        a_b=y_krig[i]+y_krig[i-1]
        sum_ave=sum_ave+0.5*hh*a_b
    sum_ave=sum_ave/h(min(x), max(x))
    plt.plot(x_krig,y_krig,linestyle='--',lw=1,label='Nugget effect = '+str(nugg))
    return sum_ave
