# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:34:57 2020

@author: BEL
"""

import io
import os
import ftplib
# from ftplib import FTP
session = ftplib.FTP('172.195.121.50')

session.login('bel','@bstc123')
files= []
session.dir(files.append)
print(files)
r = io.BytesIO()
session.retrbinary('RETR conda_List.txt', r.write)
# r.close()

info = r.getvalue().decode()
print(info)

file = open('D:/MNS/conda_List__2.txt', 'w')
file.write(info)
file.close()

# session.storbinary(cmd, fp)
# file.close()
# session.quit()