import requests
from io import BytesIO
from math import log, exp, tan, atan, ceil
from PIL import Image
import sys

#Load Google API Key
api_file = 'C:/insert/path/here/Google_API.txt'
file=open(api_file, "r")
api_key=file.read()

import urllib.request
from PIL import Image
import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import time 
import cv2

intersections = pd.read_sas('C:/insert/path/here/traffic_intersection_stack.sas7bdat')
outdir = 'C:/insert/path/here/SatelliteImages'
stu_data1 = pd.read_sas('C:/insert/path/here/geo_nc_intersections.sas7bdat')
stu_data2 = pd.read_sas('C:/insert/path/here/geo_nc_intersections_2019.sas7bdat')

data = intersections[['type', 'lat', 'long', 'angle', 'id']]
data = data.sample(frac=1) #randomly shuffle dataframe
df_subset = data.iloc[:] #take first X records
sites = df_subset.id.unique()


#for site in sites:
#    this_site = df_subset.loc[df_subset['id'] == site]
#    coords = this_site.loc[:,['id', 'lat', 'long']]
#    this_lat = coords['lat'].to_numpy()
#    this_long = coords['long'].to_numpy()
    

# circumference/radius
# One degree in radians, i.e. in the units the machine uses to store angle,
# which is always radians. For converting to and from degrees. See code for
# usage demonstration.
tau = 6.283185307179586
DEGREE = tau/360
ZOOM_OFFSET = 8
GOOGLE_MAPS_API_KEY = api_key  # set to 'your_API_key'
MAXSIZE = 640
LOGO_CUTOFF = 32
# Max width or height of a single image grabbed from Google.
# For cutting off the logos at the bottom of each of the grabbed images.  The
# logo height in pixels is assumed to be less than this amount.

def latlon2pixels(lat, lon, zoom):
    mx = lon
    my = log(tan((lat + tau/4)/2))
    res = 2**(zoom + ZOOM_OFFSET) / tau
    px = mx*res
    py = my*res
    return px, py

def pixels2latlon(px, py, zoom):
    res = 2**(zoom + ZOOM_OFFSET) / tau
    mx = px/res
    my = py/res
    lon = mx
    lat = 2*atan(exp(my)) - tau/4
    return lat, lon


def get_maps_image(site, NW_lat_long, SE_lat_long, zoom=18):
    
    ullat, ullon = NW_lat_long
    lrlat, lrlon = SE_lat_long

    # convert all these coordinates to pixels
    ulx, uly = latlon2pixels(ullat, ullon, zoom)
    lrx, lry = latlon2pixels(lrlat, lrlon, zoom)

    # calculate total pixel dimensions of final image
    dx, dy = lrx - ulx, uly - lry

    # calculate rows and columns
    cols, rows = ceil(dx/MAXSIZE), ceil(dy/MAXSIZE)

    # calculate pixel dimensions of each small image
    width = ceil(dx/cols)
    height = ceil(dy/rows)
    heightplus = height + LOGO_CUTOFF

    # assemble the image from stitched
    final = Image.new('RGB', (int(dx), int(dy)))
    for x in range(cols):
        for y in range(rows):
            dxn = width * (0.5 + x)
            dyn = height * (0.5 + y)
            latn, lonn = pixels2latlon(
                    ulx + dxn, uly - dyn - LOGO_CUTOFF/2, zoom)
            position = ','.join((str(latn/DEGREE), str(lonn/DEGREE)))
            print(x, y, position)
            urlparams = {
                    'center': position,
                    'zoom': str(zoom),
                    'size': '%dx%d' % (width, heightplus),
                    'maptype': 'terrain', #this is what you change to get different type of images
                    'sensor': 'false',
                    'scale': 1
                }
            if GOOGLE_MAPS_API_KEY is not None:
                urlparams['key'] = GOOGLE_MAPS_API_KEY

            url = 'http://maps.google.com/maps/api/staticmap'
            try:                  
                response = requests.get(url, params=urlparams)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(e)
                sys.exit(1)

            im = Image.open(BytesIO(response.content))                  
            final.paste(im, (int(x*width), int(y*height)))
            final.save("C:/insert/path/here/Terrain_Otsu_" + str(int(site)) + ".png")
            print("Terrain image patch for intersection " + str(int(site)) + " has successfully been saved")
            
    return final

kernel = np.ones((6,6),np.uint8)
kernel2 = np.ones((4,4),np.uint8)


for site in sites:
        image1 = cv2.imread("C:/insert/path/here/Terrain_" + str(int(site)) + ".png")  
        img = cv2.medianBlur(image1,5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) 
        ret, thresh2 = cv2.threshold(img,252,255,cv2.THRESH_BINARY)
        dilation = cv2.dilate(thresh2,kernel,iterations = 2)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
        closing = cv2.erode(closing, kernel2, iterations=2) 
        #cv2.imshow('image', thresh2)
        #cv2.waitKey(10)
        im = Image.fromarray(closing)
        im.save("C:/insert/path/here/Terrain_Otsu_" + str(int(site)) + ".png")
        print("Photo done!")

############################################

#36.53607,-77.7917
#36.5360125,-77.7922137
#NW_lat_long =  (36.54*DEGREE, -77.80*DEGREE)
#SE_lat_long = (36.53*DEGREE, -77.79*DEGREE)

#lat_long_offset = 0.005
#latitude, longitude = (36.50924, -77.603)

#lat_plus = round(latitude + lat_long_offset,4)
#lat_minus = round(latitude - lat_long_offset,4)
#long_plus = round(longitude + lat_long_offset,4)
#long_minus = round(longitude - lat_long_offset,4)

if __name__ == '__main__':
    
    lat_long_offset = 0.00125
    
    for site in sites: #loop over all site ids in intersection dataframe
        this_site = df_subset.loc[df_subset['id'] == site] #extract one record associated with each site id
        coords = this_site.loc[:,['id', 'lat', 'long']] 
        this_lat = coords['lat'].to_numpy() #cast lat as single element numeric array
        this_long = coords['long'].to_numpy() #cast long as single element numeric array
        
        lat_plus = round(this_lat[0] + lat_long_offset,4)
        lat_minus = round(this_lat[0] - lat_long_offset,4)
        long_plus = round(this_long[0] + lat_long_offset,4)
        long_minus = round(this_long[0] - lat_long_offset,4)
        
        NW_lat_long =  (lat_plus*DEGREE, long_minus*DEGREE)
        SE_lat_long = (lat_minus*DEGREE, long_plus*DEGREE)
    
        result = get_maps_image(site, NW_lat_long, SE_lat_long, zoom=18)
        #result.show()
        
        