ocr_list = []
# G 
if ocr_list[0] == '6':
    ocr_list[0] = 'G'
if ocr_list[1] == '6':
    ocr_list[1] = 'G'
if ocr_list[4] == '6':
    ocr_list[4] = 'G'
if ocr_list[5] == '6':
    ocr_list[5] = 'G'
# D 
if ocr_list[4] == '0':
    ocr_list[4] = 'D'
if ocr_list[5] == '0':
    ocr_list[5] = 'D'
if ocr_list[0] == '0':
    ocr_list[0] = 'D'
if ocr_list[1] == '0':
    ocr_list[1] = 'D'

if ocr_list[5] == '7' or ocr_list[5] == '1':
    ocr_list[5] = 'T'
# A 
if ocr_list[5] == '4':
    ocr_list[5] = 'A'
if ocr_list[0] =='4':
    ocr_list[0] = 'A'  
if ocr_list[1] == '4':
    ocr_list[1] = 'A'
if ocr_list[4] == '4':
    ocr_list[4] = 'A' 
# Z    
if ocr_list[5] == '2':
    ocr_list[5] = 'Z'
if ocr_list[0] =='2':
    ocr_list[0] = 'Z'
if ocr_list[1] == '2':
    ocr_list[1] = 'Z'
if ocr_list[4] == '2':
    ocr_list[4] = 'Z'
# B
if ocr_list[5] == '8':
    ocr_list[5] = 'B'
if ocr_list[0] =='8':
    ocr_list[0] = 'B'
if ocr_list[1] == '8':
    ocr_list[1] = 'B'
if ocr_list[4] == '8':
    ocr_list[4] = 'B'

if ocr_list[3] == "I" or ocr_list[3] == "T":
    ocr_list[3] = "1"
if ocr_list[2] == "I" or ocr_list[2] == "T":
    ocr_list[2] = "1"
# Z 
if ocr_list[3] == "Z":
    ocr_list[3] = "2"
if ocr_list[2] == "Z":
    ocr_list[2] = "2"
if ocr_list[6] == "Z":
    ocr_list[6] = "2"
if ocr_list[7] == "Z":
    ocr_list[7] = "2"
if ocr_list[8] == "Z":
    ocr_list[8] = "2"
if ocr_list[9] == "Z":
    ocr_list[9] = "2"

        # if ocr_list[3] == "I" or ocr_list[3] == "T" or ocr_list[2] == "I" or ocr_list[2] == "T" :    
# else:
#     if ocr_list[0].isdigit() or ocr_list[1].isdigit() or ocr_list[4].isdigit() or ocr_list[5].isdigit():
#         if ocr_list[0] == '6' or ocr_list[0] == '0' or ocr_list[0] == 'C':
#             ocr_list[0] = 'G'
#         if ocr_list[4] == '0':
#             ocr_list[4] = 'D'
#         if ocr_list[5] == '7' or ocr_list[5] == '1':
#             ocr_list[5] = 'T'
#         if ocr_list[5] == '4':
#             ocr_list[5] = 'A'
#     # if ocr_list[0] == 'N':
#     #     ocr_list[0] = 'M'
#     if not ocr_list[2].isdigit() or not ocr_list[3].isdigit() or not ocr_list[6].isdigit() or not ocr_list[7].isdigit() or not ocr_list[8].isdigit() or not ocr_list[9].isdigit():
#         if ocr_list[3] == "I" or ocr_list[3] == "T" or ocr_list[2] == "I" or ocr_list[2] == "T":
#             ocr_list[3] = "1"
# T = I or i = 1
# Z = 2
# o = 0
# Index 0  C = G
# Index 0  6 = G
# 7 = 1
# Z = 2
# A = 4
# D = 0
# G = 0
#elif ocr_list[0] == "I":