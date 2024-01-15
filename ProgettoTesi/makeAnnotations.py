"""Questa funzione serve unicamente per creare i file delle annotazioni per futuri dataset seguendo la formattazione del
file con le annotazioni originale.
Questa funzione va adeguata ad hoc per ogni nuovo dataset. Serve un file di annotazioni PER IL TRAINING
per azioni anomale  e uno per azioni normali + annotazioni per test.
per i file di training si ha:
nomevideo numerodiframeinunvideo

per il test si ha:
nomevideo numerodiframeinunvideo classevideo inizioazione fineazione inizioazione2 finoazione2
nel caso di video normali o nel caso in cui non esiste un secondo intervallo i frame di inizioazione e fineazione vanno
segnati con -1
E' IMPORTANTISSIMO CHE I VIDEO NORMALI VENGANO INSERITI IN UNA CARTELLA CHIAMATA normal.


"""


import cv2
import os

file = open('Test3_Annotation.txt', 'w')
l=os.listdir(r"C:\Users\macro\Desktop\Test2")

for video in l:
    cap= cv2.VideoCapture(os.path.join(r"C:\Users\macro\Desktop\prova" , video.replace(".txt", ".mp4")))
    print(video)
    totalframecount= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if(video[0]=="f"):
        file.writelines(os.path.join("anomaly2", video)+" "+ str(totalframecount)+" anomaly "+"1 "+str(totalframecount)+" -1"+" -1"+"\n")
    else:
        file.writelines(
            os.path.join("normal3", video) + " " + str(totalframecount) + " normal " + "-1 " + "-1" + " -1" + " -1" + "\n")
    #file.writelines(os.path.join("anomaly2", video)+" "+ str(totalframecount)+"\n")

