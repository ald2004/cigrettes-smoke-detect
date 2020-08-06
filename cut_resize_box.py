import cv2
import glob
import tqdm
allfileids=list(map(str.strip,open('allfileids','r').readlines()))
dh,dw=1080./700,1920./1430
#print(dw*(a-1)+1,dh*(b),c*dw,d*dh)
for idx,i in tqdm.tqdm(enumerate(allfileids)):
    img=cv2.imread(f'data/{i}.jpg')
    cutedimg=img[:700,490:,:]
    cv2.imwrite(f'test/{i}.jpg',cutedimg)
    anns=list(map(str.strip,open(f'data/{i}.txt','r').readlines()))
    with open(f"test/{i}.txt","w") as fid:
        for ann in anns:
            _,a,b,c,d=list(map(float,ann.split()))
            fid.write(f"0 {dw*(a-1)+1} {dh*(b)} {c*dw} {d*dh}")
            fid.write("\n")
#     if idx>20:
#         break
