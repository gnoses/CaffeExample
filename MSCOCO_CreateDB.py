# %matplotlib inline
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import PIL.Image     as Image
import PIL.ImageDraw as ImageDraw
from collections import namedtuple


# pylab.rcParams['figure.figsize'] = (10.0, 8.0)

def LoadImageList(dataDir, dataType):

    # dataDir = '/home/gnoses/DB/MS COCO'
    # dataType = 'train2014'
    annFile = '%s/annotations/instances_%s.json' % (dataDir, dataType)
    # initialize COCO api for instance annotations
    coco = COCO(annFile)
    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    # nms = [cat['name'] for cat in cats]
    # print 'COCO categories: \n\n', ' '.join(nms)
    categoryNames = {}
    for cat in cats:
        categoryNames[cat['id']] = cat['name']


        # nms = set([cat['supercategory'] for cat in cats])
        # print 'COCO supercategories: \n', ' '.join(nms)
    imgIds = coco.getImgIds()
    imgs = coco.loadImgs(imgIds)
    return coco, imgs

# load and display image
def LoadImage(coco, img,dataDir, dataType, savePath):

    # I = io.imread()
    filename = '%s/images/%s/%s'%(dataDir,dataType,img['file_name'])
    print filename
    I = Image.open(filename)

    # load and display instance annotations
    # plt.imshow(I)
    # annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    annIds = coco.getAnnIds(imgIds=img['id'])
    anns = coco.loadAnns(annIds)


    # make point list for image draw

    # labelImg = Image.new("L", (I.shape[1],I.shape[0]), 0)
    dataImg = Image.new("RGB", (640, 640), 0)
    dataImg.paste(I,(0,0))
    labelImg = Image.new("L", (640, 640), 0)
    drawer = ImageDraw.Draw( labelImg )
    Point = namedtuple('Point', ['x', 'y'])


    # polygon : list of Point
    for ann in anns:
        for id, seg in enumerate(ann['segmentation']):
            # print id, ann['category_id'], categoryNames[ann['category_id']]
            # print len(seg)
            # poly = np.array(seg).reshape((len(seg) / 2, 2))
            # try:
            if (ann['iscrowd'] == 0):
                polygon = [Point(float(seg[i]), float(seg[i+1])) for i in range(0,len(seg),2)]
            else:
                # print 'wrong', id, ann['category_id'], categoryNames[ann['category_id']]
                continue
                polygon = [Point(float(seg[i]), float(seg[i + 1])) for i in range(2, len(seg), 2)]
            # except:
                # print 'wrong',ann
                # wrong = ann
                # continue
            # print 'ok', ann
            # ok = ann
            drawer.polygon(polygon, fill=ann['category_id'])

    if (0):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(dataImg)
        # coco.showAnns(anns)

        plt.subplot(1,2,2)
        # plt.imshow(I)
        plt.imshow(labelImg)
        plt.show()
    if (1):
        saveFile = savePath + '/' + dataType + '/' + img['file_name'] + '.png'
        dataImg = dataImg.resize((320, 320), Image.BICUBIC)
        dataImg.save(saveFile)

        saveFile = savePath + '/' + dataType + 'annot/' + img['file_name'] + '.png'
        labelImg = labelImg.resize((320, 320), Image.NEAREST)
        labelImg.save(saveFile)


# get all images containing given categories, select one at random
# catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
# imgIds = coco.getImgIds(catIds=catIds );
# img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
def CreateDB(dataDir, dataType, savePath, count = None):
    # all images
    coco, imgs = LoadImageList(dataDir, dataType)
    if (count == None):
        count = len(imgs)
    for i in range(count):
        print str(i) + ' : ' + imgs[i]['file_name']
        LoadImage(coco, imgs[i],dataDir, dataType, savePath)

# 91 classes (including background : 0)
savePath = '/home/gnoses/DB/MSCOCO/Resize640x640'
CreateDB('/home/gnoses/DB/MSCOCO', 'train2014', savePath, None)
savePath = '/home/gnoses/DB/MSCOCO/Resize640x640'
CreateDB('/home/gnoses/DB/MSCOCO', 'val2014', savePath, None)

# maxWidth = 640, maxHeight = 640
# maxWidth = np.max([img1['width'] for img1 in img ])
# maxHeight = np.max([img1['height'] for img1 in img ])
