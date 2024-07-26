# 
# $kype : live:.cid.2b84562e274be2f4
# tele gram : @happydev000
# do not mention this in the free lancer site
# for your convinence.

import cv2
import os
import numpy as np
import volumentations as vol

def get_black_points(image):
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to obtain a binary image
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    
    # Find the black (0) points
    _2dPoints = np.column_stack(np.where(binary_image == 0))
    _1dPoints = np.zeros((_2dPoints.shape[0], 1))
    _3dPoints = np.concatenate((_2dPoints, _1dPoints), axis=1)
    
    return _3dPoints

def generate_image(_3dPoints):
    # Create a white image of size 200x200
    image = np.ones((200, 200), dtype=np.uint8) * 255

    # Mark black points on the image
    for point in _3dPoints:
        x, y, z = point
        if int(y) >= 0 and int(y) < 200 and int(x) >= 0 and int(x) < 200:
            image[int(y), int(x)] = 0
    
    return image

# Read the image
# image = cv2.imread('1.png')
# _3dPoints = get_black_points(image)

# volume_aug = vol.Compose([
                # vol.RotateAroundAxis3d(axis=[0, 1, 1], rotation_limit=np.pi, p=0.5)
            # ])

# m_3dPoints = volume_aug(points=_3dPoints)['points']
# m_img = generate_image(m_3dPoints)
# cv2.imwrite('2.png', m_img)

dataPoints = 0

os.mkdir('Negated_Images_Amplified')
os.chdir('Negated_Images_Amplified')
for i in range(320):
    os.mkdir('TC' + str(i + 1))
os.chdir('..')

# create 320 dataset

for i in range(40):
    for j in range(200):
        image = cv2.imread('Negated_Images/TC' + str(i + 1) + '/name_' + str(j) + '.negate.png')
        cv2.imwrite('Negated_Images_Amplified/TC' + str(i + 1) + '/name_' + str(j) + '.negate.png', image)

# rotate yz axis twice
volume_aug = vol.Compose([
                vol.RotateAroundAxis3d(axis=[0, 1, 1], rotation_limit=np.pi / 2, p=0.5),
                vol.RotateAroundAxis3d(axis=[0, 1, 1], rotation_limit=np.pi / 2, p=0.5)
            ])

for i in range(40):
    for j in range(200):
        # print(('Negated_images/TC' + str(i) + '/name_' + str(j) + '.negate.png'))
        image = cv2.imread('Negated_Images_Amplified/TC' + str(i + 1) + '/name_' + str(j) + '.negate.png')
        _3dPoints = get_black_points(image)

        m_3dPoints = volume_aug(image=_3dPoints)['image']
        m_img = generate_image(m_3dPoints)
        cv2.imwrite('Negated_Images_Amplified/TC' + str(40 + i + 1) + '/name_' + str(j) + '.negate.png', m_img)

# mirroring y axis twice
volume_aug = vol.Compose([
                vol.Flip3d(axis=[0,1,0], p=0.5),
                vol.Flip3d(axis=[0,1,0], p=0.5)
            ])
  
for i in range(80):
    for j in range(200):
        # print(('Negated_images/TC' + str(i) + '/name_' + str(j) + '.negate.png'))
        image = cv2.imread('Negated_Images_Amplified/TC' + str(i + 1) + '/name_' + str(j) + '.negate.png')
        _3dPoints = get_black_points(image)

        m_3dPoints = volume_aug(image=_3dPoints)['image']
        m_img = generate_image(m_3dPoints)
        cv2.imwrite('Negated_Images_Amplified/TC' + str(80 + i + 1) + '/name_' + str(j) + '.negate.png', m_img)

# mirroring z axis twice

volume_aug = vol.Compose([
                vol.Flip3d(axis=[0,0,1], p=0.5),
                vol.Flip3d(axis=[0,0,1], p=0.5)
            ])

for i in range(160):
    for j in range(200):
        # print(('Negated_images/TC' + str(i) + '/name_' + str(j) + '.negate.png'))
        image = cv2.imread('Negated_Images_Amplified/TC' + str(i + 1) + '/name_' + str(j) + '.negate.png')
        _3dPoints = get_black_points(image)

        m_3dPoints = volume_aug(image=_3dPoints)['image']
        m_img = generate_image(m_3dPoints)
        cv2.imwrite('Negated_Images_Amplified/TC' + str(160 + i + 1) + '/name_' + str(j) + '.negate.png', m_img)
