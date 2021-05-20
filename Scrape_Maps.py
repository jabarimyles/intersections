
import requests
from io import BytesIO
from math import log, exp, tan, atan, ceil
from PIL import Image
import sys

#Load Google API Key
api_file = 'C:/Users/jamyle/Documents/Traffic/Google_API.txt'
file=open(api_file, "r")
api_key=file.read()


# circumference/radius
# One degree in radians, i.e. in the units the machine uses to store angle,
# which is always radians. For converting to and from degrees. See code for
# usage demonstration.
tau = 6.283185307179586
DEGREE = tau/360
ZOOM_OFFSET = 8
GOOGLE_MAPS_API_KEY = 'AIzaSyCY_bXopqPxpLXOyxoarKRKqcaXM7Bn4GE'  # set to 'your_API_key'
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


def get_maps_image(NW_lat_long, SE_lat_long, zoom=18):

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
                    'maptype': 'satellite',
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
            
            #print(position)
            #print(zoom)
            #print(width)
            #print(heightplus)
            #print(latn)
            #print(lonn)

    return final

############################################

#36.53607,-77.7917
#36.5360125,-77.7922137
#NW_lat_long =  (36.54*DEGREE, -77.80*DEGREE)
#SE_lat_long = (36.53*DEGREE, -77.79*DEGREE)

if __name__ == '__main__':
    
    NW_lat_long =  (36.5425*DEGREE, -77.795*DEGREE)
    SE_lat_long = (36.5325*DEGREE, -77.785*DEGREE)

    zoom = 300  # be careful not to get too many images!

    result = get_maps_image(NW_lat_long, SE_lat_long, zoom=18)
    result.show()
