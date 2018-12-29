from tkinter import *
from tkinter import ttk
from compute import start_compute


class Row:
    '''
    每一行的参数
    mu 均值
    cov 变异系数
    dis 分布概型
    '''
    __slots__ = ('mu', 'cov', 'dis')

    def __init__(self, window, label, row, mu='', cov=''):
        mu_t = StringVar(window,value=mu)
        cov_t = StringVar(window,value=cov)
        Label(window, text=label).grid(row=row, column=0, sticky=N)
        self.mu = Entry(window, textvariable=mu_t)
        self.cov = Entry(window, textvariable=cov_t)
        self.dis = ttk.Combobox(window, state='readonly')
        self.dis['values'] = ('正态分布', '对数正态分布', '极值I型分布')
        self.dis.current(0)

        self.mu.grid(row=row, column=1, sticky=N)
        self.cov.grid(row=row, column=2, sticky=N)
        self.dis.grid(row=row, column=3, sticky=N)


class Concrete:
    '''
    混凝土数据类型
    fc 强度等级
    hc 厚度
    be 宽度
    c 保护层厚度
    Ec 弹性模量
    '''
    __slots__ = ('window','row','label','fc','hc', 'be', 'c', 'Ec')

    def __init__(self,window, row):
        self.window = window
        self.row = row
        self.label = [
            ('强度等级fc', '39.7712', '0.1578'),
            ('厚度hc', '250', '0.02'),
            ('宽度be', '2000', '0.02'),
            ('保护层厚度c', '-', '-'),
            ('弹性模量Ec', '32500', '-')
        ]

    def draw(self):
        i = 0
        self.fc = Row(self.window, self.label[i][0], self.row+i,self.label[i][1],self.label[i][2])
        i += 1
        self.hc = Row(self.window, self.label[i][0], self.row+i,self.label[i][1],self.label[i][2])
        i += 1
        self.be = Row(self.window, self.label[i][0], self.row+i,self.label[i][1],self.label[i][2])
        i += 1
        self.c = Row(self.window, self.label[i][0], self.row+i,self.label[i][1],self.label[i][2])
        i += 1
        self.Ec = Row(self.window, self.label[i][0], self.row+i,self.label[i][1],self.label[i][2])
        return i+1


class Steel:
    '''
    工字钢数据类型
    fy 强度等级
    t1 上翼缘厚度
    b1 宽度
    t2 下翼缘厚度
    b2 宽度
    tw 腹板厚度
    hw 高度
    '''
    __slots__ = ('window','row','label','fy', 't1', 'b1', 't2', 'b2', 'tw', 'hw')

    def __init__(self,window, row):
        self.window = window
        self.row = row
        self.label = [
            ('强度等级fy', '251.45', '0.081'),
            ('上翼缘厚度t1', '22', '0.02'),
            ('上翼缘宽度b1', '300', '0.02'),
            ('下翼缘厚度t2', '22', '0.02'),
            ('下翼缘宽度b2', '300', '0.02'),
            ('腹板厚度tw', '14', '0.02'),
            ('腹板高度hw', '756', '0.02')
        ]
    def draw(self):
        i = 0
        self.fy = Row(self.window, self.label[i][0], self.row+i,self.label[i][1],self.label[i][2])
        i += 1
        self.t1 = Row(self.window, self.label[i][0], self.row+i,self.label[i][1],self.label[i][2])
        i += 1
        self.b1 = Row(self.window, self.label[i][0], self.row+i,self.label[i][1],self.label[i][2])
        i += 1
        self.t2 = Row(self.window, self.label[i][0], self.row+i,self.label[i][1],self.label[i][2])
        i += 1
        self.b2 = Row(self.window, self.label[i][0], self.row+i,self.label[i][1],self.label[i][2])
        i += 1
        self.tw = Row(self.window, self.label[i][0], self.row+i,self.label[i][1],self.label[i][2])
        i += 1
        self.hw = Row(self.window, self.label[i][0], self.row+i,self.label[i][1],self.label[i][2])
        return i+1

class Stud:
    '''
    栓钉数据类型
    nr 数量
    d 直径
    fstd 栓钉抗拉强度
    '''
    __slots__ = ('window','row','label','nr', 'd','fstd')
    
    def __init__(self,window, row):
        self.window = window
        self.row = row
        self.label = [
            ('栓钉数量nr', '200', '-'),
            ('栓钉直径d', '16', '0.02'),
            ('栓钉抗拉强度fstd', '251.45', '0.081'),
        ]
    def draw(self):
        i = 0
        self.nr = Row(self.window, self.label[i][0], self.row+i,self.label[i][1],self.label[i][2])
        i += 1
        self.d = Row(self.window, self.label[i][0], self.row+i,self.label[i][1],self.label[i][2])
        i += 1
        self.fstd = Row(self.window, self.label[i][0], self.row+i,self.label[i][1],self.label[i][2])
        return i+1


class Load:
    '''
    荷载数据类型
    SG 恒载
    SQ 活载
    '''
    __slots__ = ('window','row','label','SG', 'SQ')
    
    def __init__(self,window, row):
        self.window = window
        self.row = row
        self.label = [
            ('SG', '786.64', '0.0473'),
            ('SQ', '933.48', '0.1905'),
        ]
    def draw(self):
        i = 0
        self.SG = Row(self.window, self.label[i][0], self.row+i,self.label[i][1],self.label[i][2])
        i += 1
        self.SQ = Row(self.window, self.label[i][0], self.row+i,self.label[i][1],self.label[i][2])
        return i+1

def main():
    window = Tk()
    window.title("Python")
    window.geometry('800x600')

    row = 0
    Label(window, text='TITLE').grid(row=row,sticky=N)
    row+=1
    Label(window, text='混凝土参数').grid(row=row,column=0,sticky=W)
    Label(window, text="均值").grid(row=row,column=1, sticky=W)
    Label(window, text="COV").grid(row=row,column=2, sticky=W)
    Label(window, text="分布概型").grid(row=row,column=3, sticky=W)
    row+=1

    con = Concrete(window,row)
    row += con.draw()

    Label(window, text='工字钢参数').grid(row=row,column=0,sticky=W)
    Label(window, text="均值").grid(row=row,column=1, sticky=W)
    Label(window, text="COV").grid(row=row,column=2, sticky=W)
    Label(window, text="分布概型").grid(row=row,column=3, sticky=W)
    row+=1
    ste = Steel(window,row)
    row += ste.draw()

    Label(window, text='栓钉参数').grid(row=row,column=0,sticky=W)
    Label(window, text="均值").grid(row=row,column=1, sticky=W)
    Label(window, text="COV").grid(row=row,column=2, sticky=W)
    Label(window, text="分布概型").grid(row=row,column=3, sticky=W)
    row+=1
    stu = Stud(window,row)
    row += stu.draw()

    Label(window, text='荷载参数').grid(row=row,column=0,sticky=W)
    Label(window, text="均值").grid(row=row,column=1, sticky=W)
    Label(window, text="COV").grid(row=row,column=2, sticky=W)
    Label(window, text="分布概型").grid(row=row,column=3, sticky=W)
    row+=1
    loa = Load(window,row)
    row += loa.draw()

    Button(window, text='开始计算', command=lambda:start_compute(con,ste,stu,loa)).grid(row=row)
    window.mainloop()

if __name__ == '__main__':
    main()
