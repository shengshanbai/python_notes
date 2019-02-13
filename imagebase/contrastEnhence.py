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

if __name__=='__main__':
    for (url,gama) in [('imagebase/bright.png',2.5),('imagebase/dark.png',0.5)]:
        image=cv2.imread(url)
        g=Gamma(gama)
        out=g.apply(image)
        cv2.imshow("origin",image)
        cv2.imshow("gamma",out)
        cv2.waitKey()
        cv2.destroyAllWindows()