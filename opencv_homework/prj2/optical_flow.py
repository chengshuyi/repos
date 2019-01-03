import numpy as np
from scipy import signal
import cv2 as cv
import os
def optical_flow(first_img, second_img, win):
    '''
    输入：
    first_img:第一张图片，为灰度图
    second_img:第二张图片，为灰度图
    win:窗口尺寸，用来计算LK的窗口大小
    输出：
    起始点
    终止点
    '''
    #找边缘点
    canny = cv.Canny(first_img, 255, 255)
    e_points = np.argwhere(canny!=0)      # 删除四周的边缘点
    e_points = e_points[e_points[:,0]-win>0]
    e_points = e_points[e_points[:,1]-win>0]
    e_points = e_points[e_points[:,0]+win+1<first_img.shape[0]]
    e_points = e_points[e_points[:,1]+win+1<first_img.shape[1]]
    #print('e_points shape = {}'.format(e_points.shape))
    #归一化
    f_img = np.copy(first_img)/255
    s_img = np.copy(second_img)/255
    #卷积核，用来求对x,y,t的偏导
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])
    #求二维卷积
    fx = signal.convolve2d(f_img, kernel_x, boundary='symm', mode='same')
    fy = signal.convolve2d(f_img, kernel_y, boundary='symm', mode='same')
    ft = signal.convolve2d(s_img, kernel_t, boundary='symm', mode='same') + signal.convolve2d(f_img, -1*kernel_t, boundary='symm', mode='same')
    
    idx = 0
    ret = np.zeros(e_points.shape)
    for i,j in e_points:
        Ix = fx[i-win:i+win+1, j-win:j+win+1].flatten()
        Iy = fy[i-win:i+win+1, j-win:j+win+1].flatten()
        It = ft[i-win:i+win+1, j-win:j+win+1].flatten()
        b = np.reshape(It, (It.shape[0],1))
        A = np.vstack((Ix, Iy)).T
        temp = np.zeros(2)
        #求特征值
        #if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
        temp = np.matmul(np.linalg.pinv(A), b)
        ret[idx]=temp.reshape(2)
        idx+=1
    #删除速度为0的点
    temp = np.argwhere(np.sum(np.abs(ret),axis=1)!=0)
    ret = (ret[temp[:,0],]*3).astype(int)
    e_points = e_points[temp[:,0],]
    return (e_points,e_points+ret)

def image(prefix,pictures,postfix):
    #读取第一张图片
    img_0 = cv.imread(os.path.join(prefix,pictures[0]+postfix))
    img_1 = cv.cvtColor(img_0, cv.COLOR_BGR2GRAY) 
    for i in range(1,len(pictures)):
        #读取第二张图片
        img_2 = cv.imread(os.path.join(prefix,pictures[i]+postfix))
        img_3 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)
        #两张图片计算光流
        start,end = optical_flow(img_1,img_3,3)
        #绘制红色线段
        for j in range(start.shape[0]):
            cv.line(img_0,(start[j,1],start[j,0]),(end[j,1],end[j,0]),(0,0,255),1)
        #保存图片
        cv.imwrite(os.path.join(prefix,'result-'+pictures[i-1]+postfix),img_0)
        img_0 = np.copy(img_2)
        img_1 = np.copy(img_3)

if __name__ == '__main__':
    
    print('------------------begin parse image---------------------')
    image('./eval-data/Army',['frame07','frame08','frame09','frame10','frame11','frame12','frame13','frame14'],'.png')
    print('Army done!')
    image('./eval-data/Backyard',['frame07','frame08','frame09','frame10','frame11','frame12','frame13','frame14'],'.png')
    print('Backyard done!')
    image('./eval-data/Basketball',['frame07','frame08','frame09','frame10','frame11','frame12','frame13','frame14'],'.png')
    print('Basketball done!')
    print('------------------end parse image---------------------')
    
    print()
    print()
    print('------------------begin parse video---------------------')
    cnt = 0
    cap = cv.VideoCapture('./output.avi')
    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv.VideoWriter('./result-output.avi', fourcc,30,(640,480))
    ret_1,img_0 = cap.read()
    img_1 = cv.cvtColor(img_0, cv.COLOR_BGR2GRAY)
    while True:
        ret_2,img_2 = cap.read()
        if ret_2 is True:
            img_3 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)
            start,end = optical_flow(img_1,img_3,3)
            #绘制红色线段
            for j in range(start.shape[0]):
                cv.line(img_0,(start[j,1],start[j,0]),(end[j,1],end[j,0]),(0,0,255),1)
            out.write(img_0)
            print('第{}帧处理完毕'.format(cnt))
            cnt+=1
            img_0 = np.copy(img_2)
            img_1 = np.copy(img_3)
        else:
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()
    print('------------------end parse video---------------------')