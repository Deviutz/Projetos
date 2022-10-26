import cv2
import os

#Colocar o nome do arquivo para a amostragem#
sample = cv2.imread("SOCOFing/Altered/Altered-Medium/150__M_Right_index_finger_Obl.BMP")

melhor_score = 0
filename = None
image = None
kp1, kp2, mp = None, None, None

#Verifica uma imagem relacionada#
counter = 0
for file in [file for file in os.listdir("SOCOFing/Real")][:1000]:
    if counter % 10 == 0:
        print(counter)
        print(file)
    counter += 1
    digital_image = cv2.imread("SOCOFing/Real/" + file)
    sift = cv2.SIFT_create()


    keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(digital_image, None)

    matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10},
                                    {}).knnMatch(descriptors_1, descriptors_2, k=2)

    match_points = []

    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)

    keypoints = 0
    if len(keypoints_1) < len(keypoints_2):
        keypoints = len(keypoints_1)
    else:
        keypoints = len(keypoints_2)

#Pontuação de quão similar é a imagem#
    if len(match_points) / keypoints * 100 > melhor_score:
        melhor_score = len(match_points) / keypoints * 100
        filename = file
        image = digital_image
        kp1, kp2, mp = keypoints_1, keypoints_2, match_points

#Mostrador de pontuação#
print("MELHOR MATCH: " + filename)
print("PONTUAÇÃO: " + str(melhor_score))

#Janela de comparação#
resultado = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
resultado = cv2.resize(resultado, None, fx=4, fy=4)
cv2.imshow("Resultado", resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()








