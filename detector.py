

def getIndex(img):
    return "123"

def detect(lines_and_masks):
    indexes = []
    for line in lines_and_masks[0]:
        indexes.append( getIndex(line) )
    
    return indexes