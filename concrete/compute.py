import numpy as np
from random_gen import random_gen
from config import *


def get_random_sample(obj):
    tw_mu = float(obj.mu.get())
    tw_cov = float(obj.cov.get())
    return random_gen(tw_mu,tw_mu*tw_cov,samples_num,obj.dis.get())

def start_compute(con,ste,stu,loa):
    '''
    con=concrete 混凝土
    ste=steel 工字钢
    sut=stud 栓钉
    loa=load 荷载
    '''
    print('sample num = {},years = {}'.format(samples_num,years))

    fc_t,sigmac_t = con_de_model(get_random_sample(con.fc) , float(con.fc.mu.get()), float(con.fc.cov.get()))

    tw = get_random_sample(ste.tw)
    t2 = get_random_sample(ste.t2)
    tw_t,t2_t = steel_varying_model(tw,t2,'正态分布')

    b1 = get_random_sample(ste.b1)
    d = get_random_sample(stu.d)
    p_t = stud_varying_model(b1,d,'正态分布')

def con_de_model(fc,mu,cov):
    '''
    混凝土强度退化模型
    输入：
    1. fc 强度等级
    2. mu 均值
    3. cov 变异系数
    输出:
    1. fc_t 强度等级变化
    2. sigmac_t 
    '''
    kesai_t = np.zeros(years)
    nita_t = np.zeros(years)
    sigma_t = np.zeros(samples_num)

    sigma_t = sigma_t + mu*cov
    for i in range(1,years):
        kesai_t[i] = 1.248*np.exp(-0.034*np.square((np.log(i)/np.log(np.e)-0.3468)))
        nita_t[i] = 0.0143*i+1.0624
    
    fc_t = np.mat(fc).T * np.mat(kesai_t)
    sigmac_t = np.mat(sigma_t).T*np.mat(nita_t)
    print('fc_t shape = {} sigmac_t shape = {}'.format(fc_t.shape,sigmac_t.shape))
    return fc_t,sigmac_t


def steel_varying_model(tw,t2,dis):
    '''
    钢梁-截面时变模型
    输入
    1. tw 腹板厚度
    2. t2下翼缘厚度
    3. dis分布概型
    输出
    1. tw_t
    2. t2_t
    '''
    A = random_gen(70.6,0.66*70.6,years,dis)
    B = random_gen(0.789,0.49*0.789,years,dis)
    dsc = A*np.power(years_array,B)

    tw_t = np.mat(tw).T*np.mat(np.ones((years))) - 2*np.mat(np.ones((samples_num))).T*np.mat(dsc)
    t2_t = np.mat(t2).T*np.mat(np.ones((years))) - np.mat(np.ones((samples_num))).T*np.mat(dsc)
    print('tw_t shape = {} t2_t shape = {}'.format(tw_t.shape,t2_t.shape))
    return tw_t,t2_t

def stud_varying_model(b1,d,dis):
    '''
    栓钉截面时变模型
    输入:
    1. b1 工字钢宽度
    2. d 栓钉直径
    3. dis 分布概型
    输出：
    p_t 锈蚀率
    '''
    def init_time_rust(b1,d):
        '''
        初始锈蚀时间
        输入:
        1. b1 工字钢宽度
        2. d 栓钉直径
        输出:
        Ti: 初始锈蚀时间
        '''
        Cxt = 0.05/100 #氯离子含量
        a = 0.032/100  #氯离子侵蚀参数
        k = 0.085/100  #关系系数
        Csmax = 0.85/100 #稳定值
        tcr = 10
        x = b1/2-d/2

        temp_1 = (Cxt + a*x)/k
        temp_2 = ((x-(Csmax-Cxt)/a)*a+Csmax)/k

        for i in range(samples_num):
            if temp_1[i] >= tcr:
                temp_1[i] = temp_2[i]
        print('Ti shape = {}'.format(temp_1.shape))
        return temp_1
    
    def rust_rate(d,Ti,dis):
        '''
        锈蚀率
        输入:
        1. d 栓钉直径
        2. Ti 初始锈蚀时间
        3. dis R的分布概型
        '''
        icorr = 1
        R = random_gen(6.2,6.2*1.116,samples_num,dis)
        R = np.mat(R).T * np.mat(np.ones(years))
        Ti_t = np.mat(Ti).T*np.mat(np.ones(years))
        #shape samples_num x years
        p_t = 0.0116*icorr*np.multiply(R,(np.mat(np.ones(samples_num)).T*np.mat(years_array)-Ti_t))

        D = np.mat(d).T*np.mat(np.ones(years))
        a = 2*np.multiply(p_t,np.sqrt(1-np.square(p_t/D)))
        theta_1 = 2*np.arcsin(a/D)
        theta_2 = 2*np.arcsin(a/(2*p_t))

        A1 = 1/2*(np.multiply(theta_1,np.square(D/2))-np.multiply(a,np.abs(D/2-p_t/D)))
        A2 = 1/2*(np.multiply(theta_2,np.square(p_t))-np.multiply(a,np.square(p_t)/D))

        Ar_t = np.zeros((samples_num,years))
        for i in range(samples_num):
            for j in range(years):
                if p_t[i,j] <= d[i]/np.sqrt(2):
                    Ar_t[i,j] = np.pi*np.sqrt(d[i])/4-A1[i,j] - A2[i,j]
                elif p_t[i,j] <= d[i]:
                    Ar_t[i,j] = A1[i,j] - A2[i,j]
                else:
                    Ar_t[i,j] = 0
        return 1-Ar_t/(np.mat(Ar_t[:,0]).T*np.mat(np.ones(years)))
    Ti = init_time_rust(b1,d)
    p_t = rust_rate(d,Ti,dis)
    print('p_t shape = {}'.format(p_t.shape))
    return p_t

def resist_compute(As,fy,fc,be,hc,Ec,Astd,fstd,nr,p_t,hw,t1,t2_t,fc_t,b2,tw_t):
    '''
    计算抗力
    '''
    #计算截面纵向剪力
    Vs = np.min(As*fy,fc*be*hc)
    #单个栓钉受剪承载力
    Nv = np.min(0.43*As*np.sqrt(Ec*fc),0.7*Astd*fstd)
    nf = Vs/Nv
    r_t = (0.9894-0.0334*p_t)*np.sqrt(nr/nf)
    k_t = 0.99-0.0238*p_t

    Mu_t = As_t * fy*((hw+t1+t2_t)/2+hc-(As_t*fy)/(2*fc_t*be))
    Mps = (t2_t*b2*(hw+(t1+t2_t)/2)+1/4*np.square(hw)*tw_t)*fy

    for i in range(samples_num):
        for j in range(years):
            if r_t[i,j]<1:
                Mu_t[i,j] = Mps[i,j] + k_t[i,j]*r[i,j]*(Mu_t[i,j]-Mps[i,j])
    return Mu_t

if __name__ == '__main__':
    steel_varying_model(0,0,'正态分布')




