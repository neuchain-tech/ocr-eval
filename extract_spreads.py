# extract_spreads.py
from pathlib import Path
import cv2, numpy as np
from typing import List, Tuple, Optional

def to_gray_norm(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(2.0, (8,8)).apply(g)

def adaptive_bin(gray):
    bl = cv2.GaussianBlur(gray, (5,5), 0)
    th = cv2.adaptiveThreshold(bl, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 15)
    return 255 - th  # ink=1

def content_bbox(bw):
    ys, xs = np.where(bw>0)
    if xs.size==0: return (0,0,bw.shape[1]-1,bw.shape[0]-1)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def method_projection(gray, bw):
    h,w = gray.shape
    x1,y1,x2,y2 = content_bbox(bw); roi = bw[y1:y2+1, x1:x2+1]
    if roi.size==0: return None
    v = roi.sum(axis=0).astype(np.float32)
    v = cv2.GaussianBlur(v, (1, 51 if w>2000 else 31), 0).flatten()
    c0,c1 = int(0.35*len(v)), int(0.65*len(v))
    spine = x1 + (c0 + int(np.argmin(v[c0:c1])))
    th = 0.1 * v.max()
    def bound(start, step):
        i, last = start, start
        while 0 <= i < len(v):
            if v[i] > th: last = i
            i += step
        return last
    left_end  = x1 + bound(spine-x1, -1)
    right_begin = x1 + bound(spine-x1, +1)
    return [(x1,y1,left_end,y2),(right_begin,y1,x2,y2)]

def method_hough(gray, bw):
    h,w = gray.shape
    edges = cv2.Canny(gray, 50, 150, L2gradient=True)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT,(3,7)),1)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,120,int(h*0.3),int(h*0.02))
    if lines is None: return None
    cand=[]
    for x1,y1,x2,y2 in lines[:,0]:
        dx,dy=abs(x2-x1),abs(y2-y1)
        if dy < 0.98*np.hypot(dx,dy): continue
        xm=int((x1+x2)/2)
        if int(0.3*w)<=xm<=int(0.7*w): cand.append((xm,dy))
    if not cand: return None
    xs=np.array([c[0] for c in cand]); ws=np.array([c[1] for c in cand],np.float32)
    xs,ws = xs[np.argsort(xs)], ws[np.argsort(xs)]/(ws.sum()+1e-6)
    spine = xs[min(np.searchsorted(np.cumsum(ws),0.5), len(xs)-1)]
    v = (bw>0).sum(axis=0)
    th = 0.1*v.max()
    i,left_end = spine, spine
    while i>0:
        if v[i]>th: left_end=i
        i-=1
    i,right_begin = spine, spine
    while i<w-1:
        if v[i]>th: right_begin=i
        i+=1
    x1,y1,x2,y2 = content_bbox(bw)
    return [(x1,y1,left_end,y2),(right_begin,y1,x2,y2)]

def method_two_big_contours(gray, bw):
    h,w = gray.shape
    bwc = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),1)
    cnts,_ = cv2.findContours(bwc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand=[]
    for c in cnts:
        a=cv2.contourArea(c)
        if a<0.05*h*w: continue
        x,y,ww,hh = cv2.boundingRect(c)
        ar=ww/float(hh)
        if 0.5<=ar<=2.5: cand.append((x,y,ww,hh,a))
    if len(cand)<2: return None
    best=None;score=-1
    for i in range(len(cand)):
        for j in range(i+1,len(cand)):
            x1,y1,w1,h1,a1=cand[i]; x2,y2,w2,h2,a2=cand[j]
            if min(y1+h1,y2+h2)-max(y1,y2) < 0.5*min(h1,h2): continue
            if max(abs(w1-w2)/max(w1,w2), abs(h1-h2)/max(h1,h2))>0.2: continue
            s=a1+a2
            if s>score: score=s; best=((x1,y1,x1+w1,y1+h1),(x2,y2,x2+w2,y2+h2))
    if best is None: return None
    r1,r2=best
    return [min(r1,r2,key=lambda r:r[0]), max(r1,r2,key=lambda r:r[0])]

def is_single_or_unclear(bw):
    cnts,_ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    xs=[]
    for c in cnts:
        if cv2.contourArea(c)<200: continue
        M=cv2.moments(c); 
        if M["m00"]==0: continue
        xs.append(M["m10"]/M["m00"])
    if len(xs)<10: return True
    xs=np.float32(np.array(xs)).reshape(-1,1)
    comp1,_,_=cv2.kmeans(xs,1,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,1e-4),5,cv2.KMEANS_PP_CENTERS)
    comp2,_,_=cv2.kmeans(xs,2,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,1e-4),5,cv2.KMEANS_PP_CENTERS)
    drop=(comp1-comp2)/max(comp1,1e-6)
    return drop<0.25

def consensus(rect_sets, shape):
    rs=[r for r in rect_sets if r is not None]
    if not rs: return None
    L=np.median(np.array([r[0] for r in rs],np.float32),axis=0).astype(int)
    R=np.median(np.array([r[1] for r in rs],np.float32),axis=0).astype(int)
    h,w=shape[:2]
    L=(max(0,L[0]),max(0,L[1]),min(w-1,L[2]),min(h-1,L[3]))
    R=(max(0,R[0]),max(0,R[1]),min(w-1,R[2]),min(h-1,R[3]))
    lw,lh=L[2]-L[0],L[3]-L[1]; rw,rh=R[2]-R[0],R[3]-R[1]
    if max(abs(lw-rw)/max(lw,rw,1), abs(lh-rh)/max(lh,rh,1))>0.25: return None
    if L[2]>R[0]:
        mid=(L[2]+R[0])//2
        L=(L[0],L[1],mid,L[3]); R=(mid,R[1],R[2],R[3])
    return [tuple(map(int,L)), tuple(map(int,R))]

def extract_two_pages(img: np.ndarray):
    gray=to_gray_norm(img); bw=adaptive_bin(gray)
    if is_single_or_unclear(bw): return None
    rects=[method_projection(gray,bw), method_hough(gray,bw), method_two_big_contours(gray,bw)]
    return consensus(rects, img.shape)


import cv2, sys
from pathlib import Path
from extract_spreads import extract_two_pages

p = Path(sys.argv[1])
img = cv2.imread(str(p))

if img is None:
    print(f"[ERROR] 画像が読み込めません: {p}")
    sys.exit(1)

rects = extract_two_pages(img)
if rects is None:
    print(f"[WARN] 見開き検出失敗: {p.name}")
    sys.exit(0)

(L, R) = rects
(lx1,ly1,lx2,ly2) = L
(rx1,ry1,rx2,ry2) = R

# 範囲チェック
if lx2 <= lx1 or ly2 <= ly1 or rx2 <= rx1 or ry2 <= ry1:
    print(f"[WARN] 無効な座標範囲: {p.name}")
    sys.exit(0)

left_img  = img[ly1:ly2, lx1:lx2]
right_img = img[ry1:ry2, rx1:rx2]

# 分割結果を保存
cv2.imwrite(f"{p.stem}_Left.png",  left_img)
cv2.imwrite(f"{p.stem}_Right.png", right_img)
print(f"[OK] 分割完了: {p.name}")