import os
import numpy as np
import cv2
import natsort
import matplotlib.pyplot as plt
from scipy import optimize
import datetime

class Node(object):
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value

    def printInfo(self):
        print(self.x, self.y, self.value)


def backscatter(img, percent=0.01):
    print(img.shape)
    height = img.shape[0]
    width = int(img.shape[1]/10)
    size = height * width

    nodes1 = []
    nodes2 = []
    nodes3 = []
    nodes4 = []
    nodes5 = []
    nodes6 = []
    nodes7 = []
    nodes8 = []
    nodes9 = []
    nodes10 = []
    
    for i in range(0, height):
        for j in range(0, width):
            oneNode = Node(i, j, sum(tuple(img[i, j])))
            nodes1.append(oneNode)
        for j in range(width, width*2):
            oneNode = Node(i, j, sum(tuple(img[i, j])))
            nodes2.append(oneNode)
        for j in range(width*2, width*3):
            oneNode = Node(i, j, sum(tuple(img[i, j])))
            nodes3.append(oneNode)
        for j in range(width*3, width*4):
            oneNode = Node(i, j, sum(tuple(img[i, j])))
            nodes4.append(oneNode)
        for j in range(width*4, width*5):
            oneNode = Node(i, j, sum(tuple(img[i, j])))
            nodes5.append(oneNode)
        for j in range(width*5, width*6):
            oneNode = Node(i, j, sum(tuple(img[i, j])))
            nodes6.append(oneNode)
        for j in range(width*6, width*7):
            oneNode = Node(i, j, sum(tuple(img[i, j])))
            nodes7.append(oneNode)
        for j in range(width*7, width*8):
            oneNode = Node(i, j, sum(tuple(img[i, j])))
            nodes8.append(oneNode)
        for j in range(width*8, width*9):
            oneNode = Node(i, j, sum(tuple(img[i, j])))
            nodes9.append(oneNode)
        for j in range(width*9, width*10):
            oneNode = Node(i, j, sum(tuple(img[i, j])))
            nodes10.append(oneNode)
    #print(sum(tuple(img[1, 1])))
    #print(tuple(img[1, 1]))        
    #print(nodes1[1])            
    
    nodes1 = sorted(nodes1, key=lambda node: node.value, reverse=False)
    nodes2 = sorted(nodes2, key=lambda node: node.value, reverse=False)
    nodes3 = sorted(nodes3, key=lambda node: node.value, reverse=False)
    nodes4 = sorted(nodes4, key=lambda node: node.value, reverse=False)
    nodes5 = sorted(nodes5, key=lambda node: node.value, reverse=False)
    nodes6 = sorted(nodes6, key=lambda node: node.value, reverse=False)
    nodes7 = sorted(nodes7, key=lambda node: node.value, reverse=False)
    nodes8 = sorted(nodes8, key=lambda node: node.value, reverse=False)
    nodes9 = sorted(nodes9, key=lambda node: node.value, reverse=False)
    nodes10 = sorted(nodes10, key=lambda node: node.value, reverse=False)
    #print(type(img[nodes1[i].x, nodes1[i].y]/255))
 
    imgR = []
    imgG = []
    imgB = []
    depth = []
    for i in range(0, int(percent * size)):
        imgB.append(img[nodes1[i].x, nodes1[i].y, 0])        
        imgB.append(img[nodes2[i].x, nodes2[i].y, 0])
        imgB.append(img[nodes3[i].x, nodes3[i].y, 0])
        imgB.append(img[nodes4[i].x, nodes4[i].y, 0])
        imgB.append(img[nodes5[i].x, nodes5[i].y, 0])
        imgB.append(img[nodes6[i].x, nodes6[i].y, 0])
        imgB.append(img[nodes7[i].x, nodes7[i].y, 0])
        imgB.append(img[nodes8[i].x, nodes8[i].y, 0])
        imgB.append(img[nodes9[i].x, nodes9[i].y, 0])
        imgB.append(img[nodes10[i].x, nodes10[i].y, 0])
        imgG.append(img[nodes1[i].x, nodes1[i].y, 1])        
        imgG.append(img[nodes2[i].x, nodes2[i].y, 1])
        imgG.append(img[nodes3[i].x, nodes3[i].y, 1])
        imgG.append(img[nodes4[i].x, nodes4[i].y, 1])
        imgG.append(img[nodes5[i].x, nodes5[i].y, 1])
        imgG.append(img[nodes6[i].x, nodes6[i].y, 1])
        imgG.append(img[nodes7[i].x, nodes7[i].y, 1])
        imgG.append(img[nodes8[i].x, nodes8[i].y, 1])
        imgG.append(img[nodes9[i].x, nodes9[i].y, 1])
        imgG.append(img[nodes10[i].x, nodes10[i].y, 1])
        imgR.append(img[nodes1[i].x, nodes1[i].y, 2])        
        imgR.append(img[nodes2[i].x, nodes2[i].y, 2])
        imgR.append(img[nodes3[i].x, nodes3[i].y, 2])
        imgR.append(img[nodes4[i].x, nodes4[i].y, 2])
        imgR.append(img[nodes5[i].x, nodes5[i].y, 2])
        imgR.append(img[nodes6[i].x, nodes6[i].y, 2])
        imgR.append(img[nodes7[i].x, nodes7[i].y, 2])
        imgR.append(img[nodes8[i].x, nodes8[i].y, 2])
        imgR.append(img[nodes9[i].x, nodes9[i].y, 2])
        imgR.append(img[nodes10[i].x, nodes10[i].y, 2])
        depth.append(depth_map[nodes1[i].x, nodes1[i].y])
        depth.append(depth_map[nodes2[i].x, nodes2[i].y])
        depth.append(depth_map[nodes3[i].x, nodes3[i].y])
        depth.append(depth_map[nodes4[i].x, nodes4[i].y])
        depth.append(depth_map[nodes5[i].x, nodes5[i].y])
        depth.append(depth_map[nodes6[i].x, nodes6[i].y])
        depth.append(depth_map[nodes7[i].x, nodes7[i].y])
        depth.append(depth_map[nodes8[i].x, nodes8[i].y])
        depth.append(depth_map[nodes9[i].x, nodes9[i].y])
        depth.append(depth_map[nodes10[i].x, nodes10[i].y]) 
    imgB = np.array(imgB)/255
    imgG = np.array(imgG)/255
    imgR = np.array(imgR)/255
    depth = np.array(depth)/255
    '''
    plt.figure(figsize=(8, 6))
    plt.scatter(depth, imgG)
    p0 = [1,5,1,5]
    s="Test the number of iteration"
    Para=optimize.leastsq(test_err,p0,args=(depth,imgG,s),maxfev = 10000)
    a,b,c,d = Para[0]
    print("a=",a,'\n',"b=",b,"c=",c,"d=",d)
    plt.figure(figsize=(8,6))
    plt.scatter(depth,imgG,color="red",label="Color Point",linewidth=0.1)
    x=np.linspace(0,1,1000)
    y=a*(1-np.exp(-b*x))+c*np.exp(-d)
    plt.plot(x,y,color="orange",label="Fitting Curve",linewidth=2)
    plt.legend()
    plt.show()
    '''
    ab,bb,cb,db = nls(depth,imgB)
    ag,bg,cg,dg = nls(depth,imgG)
    ar,br,cr,dr = nls(depth,imgR)
    '''
    plt.scatter(depth,imgB,color="red",label="Color Point",s=0.1)
    x=np.linspace(0.4,1)
    y=ab*(1-np.exp(-bb*x))+cb*np.exp(-db)
    plt.plot(x,y,color="orange",label="Fitting Curve",linewidth=2)
    plt.legend()
    plt.show()
    '''
    bsrm = np.zeros(img.shape)
    bsrm = np.float64(bsrm)
    #print(np.min(depth_map)/255)
    BCtestb = np.zeros(depth_map.shape)
    BCtestg = np.zeros(depth_map.shape)
    BCtestr = np.zeros(depth_map.shape)
    BCtestb[:,:] = (ab*(1-np.exp(-(bb)*(depth_map[:,:]/255)))+cb*np.exp(-(db)))
    BCtestg[:,:] = (ag*(1-np.exp(-(bg)*(depth_map[:,:]/255)))+cg*np.exp(-(dg)))
    BCtestr[:,:] = (ar*(1-np.exp(-(br)*(depth_map[:,:]/255)))+cr*np.exp(-(dr)))
    #print(np.where(BCtestb==np.min(BCtestb)))
    print(depth_map[0,0]/255)
    for i in range(0,3):
        
        if i == 0:
            bsrm[:,:,i] = img[:,:,i]/255 - (ab*(1-np.exp(-(bb)*(depth_map[:,:]/255)))+cb*np.exp(-(db)))     
        if i == 1:
            bsrm[:,:,i] = img[:,:,i]/255 - (ag*(1-np.exp(-(bg)*(depth_map[:,:]/255)))+cg*np.exp(-(dg)))
        if i == 2:
            bsrm[:,:,i] = img[:,:,i]/255 - (ar*(1-np.exp(-(br)*(depth_map[:,:]/255)))+cr*np.exp(-(dr)))
    #print(bsrm)
    bsrm = np.array(bsrm)*255
    print(np.where(bsrm<0))
    print(img[0,0,2]/255)
    #bsrm = np.clip(bsrm,0,255)
    #bsrm = np.uint8(bsrm)
    #print(bsrm)
    cv2.imwrite('testre0209.jpg',bsrm)
    #print(np.min(bsrm))
    #bc = 
    
    return bsrm

def nls(depth,img):
    p0 = [1,5,1,5]
    s="Test the number of iteration"
    Para=optimize.leastsq(test_err,p0,args=(depth,img,s),maxfev = 10000)
    a,b,c,d = Para[0]
    #print("a=",a,'\n',"b=",b,"c=",c,"d=",d)
    return a,b,c,d

def nls2(depth,img):
    p1 = [1,-1,1,-1]
    s="Test the number of iteration"
    Para=optimize.leastsq(test_err2,p1,args=(depth,img,s),maxfev = 10000)
    a,b,c,d = Para[0]
    return a,b,c,d

def test_err2(p, x, y, s):
    return fit1(p,x)-y
    
def test_func(p, x):
    a,b,c,d = p
    return a*(1-np.exp(-(b)*x))+c*np.exp(-(d))

def test_err(p, x, y, s):
    #print(s)
    return test_func(p, x)-y

def fit(x,a,b,c,d):
    return a * np.exp(b * x) + c * np.exp(d * x)
def fit1(p,x):
    a,b,c,d=p
    return a * np.exp(b * x) + c * np.exp(d * x)

def direct_signal(img,ill,depths):
    #print(np.min(img),np.max(img))
    roughDC = np.zeros(img.shape)
    #roughDC = np.float64(roughDC)
    #print(roughDC)
    ill = ill/255
    #print(ill)
    depths = (depths/255)*30
    #print(depths)
    #ill = np.float64(ill)
    #print(ill)
    #depths = np.float64(depths)
    #print(depths)
    
    
    for i in range(0,3):
        if i == 0:
            roughDC[:,:,i] = -np.log(ill[:,:,i])/depths[:,:]
        if i == 1:
            roughDC[:,:,i] = -np.log(ill[:,:,i])/depths[:,:]
        if i == 2:
            roughDC[:,:,i] = -np.log(ill[:,:,i])/depths[:,:]
    #roughDb = np.zeros[img.shape[0],img.shape[1]]
    (roughDb,roughDg,roughDr) = cv2.split(roughDC)
    #roughDb = roughDC[:,:,0]        
    depth1d = depths.flatten()
    roughDg1d = roughDg.flatten()

    test=np.argwhere(np.isinf(roughDg1d))
    
    
    test = test.flatten()
    #print(np.max(roughDg1d),np.min(roughDg1d))
    #print(test)
    roughDg1d = np.delete(roughDg1d,test)
    depth1d = np.delete(depth1d,test)
    #plt.scatter(depth1d,roughDb1d,color="red",s=0.1)
    #plt.show()
    #print(roughDr1d.shape)
    #print(roughDr1d.shape)
    #print(depth1d)
    #print(type(depth1d))
    #plt.scatter(depth1d,roughDr1d)
    #print(roughDr1d[0],depth1d[0])
    '''
    delt = []
    for i in range(roughDr1d.shape[0]):
        if roughDr1d[i] > 0.19 and 20<depth1d[i]:
            delt = np.append(delt,i)
    delt = delt.astype(int)
    print(delt.shape)
    roughDr1d = np.delete(roughDr1d,delt)
    depth1d = np.delete(depth1d,delt)
    print(roughDr1d.shape)
    '''

    '''
    p0 = [1,-1,1,-1]
    s="Test the number of iteration"
    Para=optimize.leastsq(test_err2,p0,args=(depth1d,roughDr1d,s))
    a,b,c,d = Para[0]
    print("a=",a,'\n',"b=",b,"c=",c,"d=",d)
    plt.figure(figsize=(8,6))
    plt.scatter(depth1d,roughDr1d,color="red",s=0.1)
    x=np.linspace(15,30)
    y=a * np.exp(b * x) + c * np.exp(d * x)
    plt.plot(x,y,color="orange",linewidth=2)
    #plt.legend()
    plt.show()
    '''
    '''
    aar = 13.150756479344743
    bbr = -0.7867214770755953
    ccr = 0.3516722662800257
    ddr = -0.03542441682996397
    aab = 52.016718157470265
    bbb = -1.3328650557402235
    ccb = 0.04265857011788078
    ddb = -4.095876900067823e-31
    aag = 58.51971316729428
    bbg = -1.338099348754493
    ccg = 0.047919163618505387
    ddg = -5.228101106579538e-37
    '''
    
    aar = 0.10551773436295755
    bbr = -8.089204799801317e-10
    ccr = 78281434077916.39
    ddr = -2.337517254290636
    aab = 1.036472773743879
    bbb = -0.657541188189825
    ccb = 0.2653342919673902
    ddb = -0.20791393162982874
    aag = 0.3472418332457873
    bbg = -0.5682546593444208
    ccg = 0.04725885290112048
    ddg = -0.09209111289125627
    
    #print(roughDb1d)
    #ta,tb,tc,td = nls2(depth1d,roughDr1d)
    #print(ta,tb,tc,td)
    #x=np.linspace(0.4,1)
    #y=ta * np.exp(tb * x) + tc * np.exp(td * x)
    #plt.plot(x,y,color="orange",label="Fitting Curve",linewidth=2)
    #plt.legend()
    #plt.show()
    
    #test = []
    #print(np.argwhere(np.isnan(roughDr1d)))
    #print(test.shape)
    #print(roughDr1d.shape)
    #roughDr1d = np.nan_to_num(roughDr1d,posinf=9)
    
    popt, pcov = optimize.curve_fit(fit, depth1d, roughDg1d,bounds=([0,-np.inf,0,-np.inf],[np.inf,0,np.inf,0]))
    a = popt[0]
    b = popt[1]
    c = popt[2]
    d = popt[3]
    print(a,b,c,d)
    x=np.linspace(10,30)
    print(a * np.exp(b * 12) + c * np.exp(d * 12))
    yvals=fit(x,a,b,c,d)
    plt.figure(figsize=(8, 6))
    plt.plot(x, yvals, 'orange',label='polyfit values',linewidth=3)
    plt.scatter(depth1d,roughDg1d,color='b',s=0.1)
    plt.show()  
    
    Jc = np.zeros(img.shape)
    img = img/255
    #print(np.max(img))
    for i in range(0,3):
        if i == 0:
            Jc[:,:,i] = img[:,:,i]*np.exp(1/(1+np.exp(-(aab*np.exp(bbb*depths[:,:])+ccb*np.exp(ddb*depths[:,:]))*depths[:,:])))#(1/(1+np.exp((aab*np.exp(bbb*depths[:,:])+ccb*np.exp(ddb*depths[:,:]))*depths[:,:])))#np.exp((aab*np.exp(bbb*depths)+ccb*np.exp(ddb*depths))*depths)
        if i == 1:
            Jc[:,:,i] = img[:,:,i]*np.exp(1/(1+np.exp(-(aag*np.exp(bbg*depths[:,:])+ccg*np.exp(ddg*depths[:,:]))*depths[:,:])))#(1/(1+np.exp((aag*np.exp(bbg*depths[:,:])+ccg*np.exp(ddg*depths[:,:]))*depths[:,:])))#np.exp((aag*np.exp(bbg*depths)+ccg*np.exp(ddg*depths))*depths)
        if i == 2:
            Jc[:,:,i] = img[:,:,i]*np.exp(1/(1+np.exp(-(aar*np.exp(bbr*depths[:,:])+ccr*np.exp(ddr*depths[:,:]))*depths[:,:])))#(1/(1+np.exp((aar*np.exp(bbr*depths[:,:])+ccr*np.exp(ddr*depths[:,:]))*depths[:,:])))#np.exp((aar*np.exp(bbr*depths)+ccr*np.exp(ddr*depths))*depths)
    zzz = np.zeros(depths.shape)
    xxx = np.zeros(depths.shape)
    yyy = np.zeros(depths.shape)
    zzz[:,:] = np.exp(np.log((aab*np.exp(bbb*depths[:,:])+ccb*np.exp(ddb*depths[:,:]))*depths[:,:]))
    xxx[:,:] = np.exp(np.log((aag*np.exp(bbg*depths[:,:])+ccg*np.exp(ddg*depths[:,:]))*depths[:,:]))
    yyy[:,:] = np.exp(np.log((aar*np.exp(bbr*depths[:,:])+ccr*np.exp(ddr*depths[:,:]))*depths[:,:]))
    print(np.mean(zzz),np.mean(xxx),np.mean(yyy))
    Jc = (Jc/np.max(Jc))*255
    print(np.max(Jc))
    #Jc = np.clip(Jc,0,160)
    #Jc = Jc*0.75
    #Jc = np.clip(Jc,0,255)
    
    #Jc = Jc+(1.8445552180408813)
    #Jc = Jc*255
    
    #Jc = np.clip(Jc,0,255)
    cv2.imwrite("Jc0209.jpg",Jc)
    return roughDC
    
np.seterr(over='ignore')
if __name__ == '__main__':
    pass
starttime = datetime.datetime.now()
folder = "D:/DCP"
path = folder + "/InputImages"
files = os.listdir(path)
files =  natsort.natsorted(files)

for i in range(len(files)):
    file = files[i]
    filepath = path + "/" + file
    prefix = file.split('.')[0]
    if os.path.isfile(filepath):
        print('********    file   ********',file)
        img = cv2.imread(folder +'/InputImages/' + file)
        depth_map = cv2.imread('D:/DCP/InputDepth/test242525.jpg', cv2.IMREAD_GRAYSCALE)
        estill = cv2.imread('D:/DCP/LSAregg2626.jpg')
        testDC = cv2.imread('D:/DCP/InputImages/RGT_0209.jpg')
    DC = backscatter(img, 0.01)
    #print(DC)
    JC = direct_signal(testDC,estill,depth_map)
        #cv2.imwrite('OutputImages/' + prefix + '_dark.jpg', imgDark)
        #transmission, sceneRadiance = getRecoverScene(img)
        #cv2.imwrite('OutputImages/' + prefix + '_DCP_TM.jpg', np.uint8(transmission * 255))
        #cv2.imwrite('OutputImages/' + prefix + '_DCP.jpg', sceneRadiance)
Endtime = datetime.datetime.now()
Time = Endtime - starttime
print('Time', Time)