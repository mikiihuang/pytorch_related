import re

class Reader:
    def __init__(self,filename,needFresh = True):

        with open(filename,encoding="utf-8") as f:
            self.fileText = f.readlines()
        if needFresh == True:
            freshData = self.refresh_data(self.fileText)
        # for data in


    def refresh_data(self):
        pass


