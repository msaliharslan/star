# -*- coding: utf-8 -*-

import glob
import os

os.chdir("../")

fileNames = glob.glob("Records/fish*.bag" )


targetRecors = open( "Records/targetRecords.txt", "w" )
targetRecors.truncate(0)  # remove the content of the file

for line in fileNames:
    targetRecors.write(line.lstrip('Records').lstrip('/') + '\n' )

targetRecors.close()

