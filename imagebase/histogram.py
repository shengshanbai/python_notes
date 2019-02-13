import cv2
import numpy
import copy

"""
图像直方图相关函数
"""
def calcHist(image):
    """
    计算图像的归一化直方图，如果图像有多个通道，对每个通道单独计算，
    """
    w=0
    h=0
    d=1
    if len(image.shape) == 2:
        w,h=image.shape
    elif len(image.shape) == 3:
        w,h,d=image.shape
    hAll=tuple(([0,]*256 for i in range(d)))
    for i,element in enumerate(image.flat):
        hAll[i%d][element]+=1/(w*h)
    return hAll

def showHist(hists,color):
    """
    显示图像直方图
    """
    for i,hist in enumerate(hists):
        histImg = numpy.zeros([256,256,3], numpy.uint8)
        maxValue=max(hist)
        hpt = int(0.9* 256)
        for h in range(256):  
            intensity = int(hist[h]*hpt/maxValue)
            cv2.line(histImg,(h,256), (h,256-intensity), color)
        cv2.imshow("hist"+str(i),histImg)
    cv2.waitKey()
    cv2.destroyAllWindows()

def equalizeHist(image):
    """
    直方图均衡算法
    """
    hists = calcHist(image)
    table = list()
    for hist in hists:
        cHist=list([round(255*sum(hist[0:i])) for i in range(len(hist))])
        table.append(cHist)
    shape=image.shape
    d=len(hists)
    out=numpy.array([table[i%d][x] for i,x in enumerate(image.flat)],dtype=numpy.uint8)
    out=numpy.reshape(out,shape)
    return out


if __name__=='__main__':
    image = cv2.imread('imagebase/dark.png')
    #hist=calcHist(image)
    #showHist(hist,(0,255,255))
    out=equalizeHist(image)
    cv2.imshow("origin",image)
    cv2.imshow("hequlize",out)
    cv2.waitKey()
    cv2.destroyAllWindows()
