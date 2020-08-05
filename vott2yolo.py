import os,json,glob
from shutil import copyfile
import tqdm
#copyfile(src, dst)
allannfiles=glob.glob('/data01/smoking/A_ano/a/jw/output/*.json')
def convert_xminymin_xcenterycenter(h, w, xmin, ymin, xmax, ymax):
    # < x_center > < y_center > < width > < height > - float values relative to width and height of image, it can  be  equal from (0.0 to 1.0]
    dw = 1. / (float(w))
    dh = 1. / (float(h))
    x = (xmin + xmax) / 2.0

    y = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    x = round(x * dw, 6)
    w = round(w * dw, 6)
    y = round(y * dh, 6)
    h = round(h * dh, 6)
    #     return x, y, w, h
    return f'{x} {y} {w} {h}'
annsdir='/data01/yuanpu/smoke/data/anns'
picturesdir='/data01/smoking/A_ano/a/jw'
for idx,i in tqdm.tqdm(enumerate(allannfiles)):
    obj=json.loads(''.join(list(map(str.strip,open(i,'r').readlines()))))
    annfilename=os.path.splitext(obj['asset']['name'])[0]+'.txt'
    h,w=obj['asset']['size']['height'],obj['asset']['size']['width']
    with open(os.path.join(annsdir,annfilename),'w') as fid:
        for xx in obj['regions']:
            a,b,c,d=map(float,xx['boundingBox'].values())
            obbox=convert_xminymin_xcenterycenter(h,w,c,d,c+b,d+a)
            fid.write(f"0 {obbox}\n")
    copyfile(os.path.join(picturesdir,obj['asset']['name']),os.path.join(annsdir,obj['asset']['name']),)
#     if idx>10:
#         break
