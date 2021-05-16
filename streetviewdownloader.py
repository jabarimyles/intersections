# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 08:12:27 2019

@author: jamyle
"""

import urllib, os
import pandas as pd
from PIL import Image

intersections = pd.read_sas('C:/Users/jamyle/Documents/Traffic/traffic_intersection_stack.sas7bdat')
outdir = 'C:/Users/jamyle/OneDrive - SAS/Copy of jamyle/Documents/Traffic/StreetViewImages/'

data_sub = intersections[['type', 'lat', 'long', 'angle', 'id' ]]

################################################################################################################################

key = "&key=" + "" #got banned after ~100 requests with no key

def GetStreet(Add,SaveLoc):
  base = "https://maps.googleapis.com/maps/api/streetview?size=1200x800&location="
  MyUrl = base + urllib.parse.quote_plus(Add) + key #added url encoding
  fi = Add + ".jpg"
  urllib.request.urlretrieve(MyUrl, os.path.join(SaveLoc,fi))

Tests = ["35.2734,-82.6622"]

for i in Tests:
  GetStreet(Add=i,SaveLoc=outdir)
  
################################################################################################################################
  
  
  
import google_streetview.api
for i in range(653, len(data_sub)):
    # Create a new image of the size require
        j = 0
        map_img = Image.new('RGB', (1200, 640))
    
        headings = ['0','90','180', '270']
    
        lat = data_sub.ix[i, 'lat']
    
        long = data_sub.ix[i, 'long']
    
        location = str(lat) + "," + str(long)
    
        for heading in headings:
            
            params = [{
	                'size': '300x640', # max 640x640 pixels
	                'location': location,
	                'heading': heading,
	                'pitch': '0',
	                'key': 'AIzaSyAPn85js-TKsUF5Cx4iO3zKW6K14hQ_xnA',
                    'fov': '90'
                   }]
    
    
            results = google_streetview.api.results(params)
        
            results.download_links('C:\\Users\\jamyle\\Documents\\Traffic\\StreetImages')
        
            img = Image.open('C:/Users/jamyle/Documents/Traffic/StreetImages/gsv_0.jpg')

            map_img.paste(img, (j * 300, 0))
            print(j)
            j+=1

        map_img.save(outdir +'Streetview_' + str(data_sub.ix[i,'id']) + '.png' )
        print("Picture saved!")

print("All pictures have been saved!")