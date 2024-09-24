import pandas as pd
import os

images = os.listdir("../../images")

s0_df = pd.read_csv("../../s0_rel.csv")
s1_df = pd.read_csv("../../s1_rel.csv")

found_images = []


for i in range(len(s0_df)):
    if i % 100 == 0: print(i)

    for image in images:
        if image == s0_df.loc[i]["IMAGE1"] or image == s0_df.loc[i]["IMAGE2"]:
            found_images.append(s0_df.loc[i])
            images.pop(images.index(image))
            break

for i in range(len(s1_df)):
    if i % 100 == 0: print(i)
    
    for image in images:
        if image == s1_df.loc[i]["IMAGE1"] or image == s1_df.loc[i]["IMAGE2"]:
            found_images.append(s1_df.loc[i])
            images.pop(images.index(image))
            break

print(len(images))
print(len(found_images))
print(found_images[:10])


