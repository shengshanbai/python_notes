import cv2
import numpy

"""
主要是一些增强对比对的算法
"""
class Gamma:
    """
    gamma变换，也叫幂律变换，gamma大于1时，增加过亮区域的对比度，小于1时
    增加暗区域的对比度,公式:s=cr^gama
    """
    def __init__(self,gamma):
        self.gamma=gamma
        self.c=255/(pow(255,gamma))
        self.table=[round(self.c*pow(i,gamma)) for i in range(256)]

    def apply(self,image):
        shape=image.shape
        out=numpy.array([self.table[x] for x in image.flat],dtype=numpy.uint8)
        out=numpy.reshape(out,shape)
        return out

class PieLineTrans:
    """
    分段线性变换，有2个点(r1,s1),(r2,s2)，分别从(0-r1),(r1-r2),(r2-max)进行三段线性变换，
    可以看做是对gamma变换的一种近似，更好的根据需要调整值
    """
    def __init__(self,r1,s1,r2,s2):
        self.table=[round(x*s1/r1) for x in range(r1)]
        self.table.extend([round((x-r1)*(s2-s1)/(r2-r1)+s1) for x in range(r1,r2)])
        self.table.extend([round((x-r2)*(255-s2)/(255-r2)+s2) for x in range(r2,256)])
    
    def apply(self,image):
        shape=image.shape
        out=numpy.array([self.table[x] for x in image.flat],dtype=numpy.uint8)
        out=numpy.reshape(out,shape)
        return out

def testGamma():
    for (url,gama) in [('imagebase/bright.png',2.5),('imagebase/dark.png',0.5)]:
        image=cv2.imread(url)
        g=Gamma(gama)
        out=g.apply(image)
        cv2.imshow("origin",image)
        cv2.imshow("gamma",out)
        cv2.waitKey()
        cv2.destroyAllWindows()

def testLineTrans():
    for url in ['imagebase/bright.png','imagebase/dark.png']:
        lineTrans=PieLineTrans(70,20,186,230)
        image=cv2.imread(url)
        out=lineTrans.apply(image)
        cv2.imshow("origin",image)
        cv2.imshow("gamma",out)
        cv2.waitKey()
        cv2.destroyAllWindows()

if __name__=='__main__':
    #testGamma()
    testLineTrans()
