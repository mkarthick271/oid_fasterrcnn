import pandas as pd
f=pd.read_csv("train-images-boxable-with-rotation.csv")
keep_col = ['ImageID','Subset','OriginalURL','OriginalLandingURL','License','AuthorProfileURL','Author','OriginalSize','OriginalMD5','Thumbnail300KURL','Rotation']
new_f = f[keep_col]
new_f.to_csv("newFile.csv", index=False)
