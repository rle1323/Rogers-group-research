from import_images import grab_images
import numpy as np

images = grab_images('images_data.txt')

def generate_labels(features):
    all_labels = []
    for _ in range(features.shape[0]):
        labels = []
        for num, image in enumerate(features[0], start=2):
            stress = float(num)/2
            if stress < 50: labels.append(0)
            elif stress >= 50 and stress < 100: labels.append(1)
            elif stress >= 100 and stress < 150: labels.append(2)
            elif stress >= 150 and stress < 200: labels.append(3)
            elif stress >= 200 and stress < 250: labels.append(4) 
            elif stress >= 250 and stress < 300: labels.append(5) 
            elif stress >= 300 and stress < 350: labels.append(6) 
            elif stress >= 350 and stress < 400: labels.append(7) 
            elif stress >= 400 and stress < 450: labels.append(8) 
            elif stress >= 450 and stress < 501: labels.append(9)
            
        labels = np.asarray(labels)
        all_labels.append(labels)
    all_labels = np.asarray(all_labels)
    return all_labels



'''
  if stress < 25: labels.append(0)
            elif stress >= 25 and stress < 50: labels.append(1)
            elif stress >= 50 and stress < 75: labels.append(2)
            elif stress >= 75 and stress < 100: labels.append(3)
            elif stress >= 100 and stress < 125: labels.append(4) 
            elif stress >= 125 and stress < 150: labels.append(5) 
            elif stress >= 150 and stress < 175: labels.append(6) 
            elif stress >= 175 and stress < 200: labels.append(7) 
            elif stress >= 200 and stress < 225: labels.append(8) 
            elif stress >= 225 and stress < 250: labels.append(9)
            elif stress >= 250 and stress < 275: labels.append(10)
            elif stress >= 275 and stress < 300: labels.append(11)
            elif stress >= 300 and stress < 325: labels.append(12)
            elif stress >= 325 and stress < 350: labels.append(13) 
            elif stress >= 350 and stress < 375: labels.append(14)
            elif stress >= 375 and stress < 400: labels.append(15) 
            elif stress >= 400 and stress < 425: labels.append(16) 
            elif stress >= 425 and stress < 450: labels.append(17) 
            elif stress >= 450 and stress < 475: labels.append(18) 
            elif stress >= 475 and stress < 501: labels.append(19)


            if stress < 50: labels.append(0)
            elif stress >= 50 and stress < 100: labels.append(1)
            elif stress >= 100 and stress < 150: labels.append(2)
            elif stress >= 150 and stress < 200: labels.append(3)
            elif stress >= 200 and stress < 250: labels.append(4) 
            elif stress >= 250 and stress < 300: labels.append(5) 
            elif stress >= 300 and stress < 350: labels.append(6) 
            elif stress >= 350 and stress < 400: labels.append(7) 
            elif stress >= 400 and stress < 450: labels.append(8) 
            elif stress >= 450 and stress < 501: labels.append(9)
'''