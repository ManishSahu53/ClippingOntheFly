from flask import Flask, request, jsonify, Response, send_file, send_from_directory
import base64
from requests.utils import requote_uri
"""app.main: handle request for lambda-tiler"""

import re
import json

import numpy as np
from flask_compress import Compress
from flask_cors import CORS

from rio_tiler import main
from rio_tiler.utils import (array_to_image,
                             linear_rescale,
                             get_colormap,
                             expression,
                             mapzen_elevation_rgb)

import time
import gzip
import geoconvert
# from lambda_proxy.proxy import API

# Cliping
import rasterio
import pyproj
import rasterio.mask
import fiona

from PIL import Image
from io import BytesIO, StringIO
import base64
import numpy as np
from rio_tiler.errors import (RioTilerError,
                              InvalidFormat,
                              InvalidLandsatSceneId,
                              InvalidSentinelSceneId,
                              InvalidCBERSSceneId)


def reshape_as_image(arr):
    """Returns the source array reshaped into the order
    expected by image processing and visualization software
    (matplotlib, scikit-image, etc)
    by swapping the axes order from (bands, rows, columns)
    to (rows, columns, bands)

    Parameters
    ----------
    arr : array-like of shape (bands, rows, columns)
        image to reshape
    """
    # swap the axes order from (bands, rows, columns) to (rows, columns, bands)
    im = np.ma.transpose(arr, [1, 2, 0])
    return im

def b64_encode_img(img, tileformat):
    """Convert a Pillow image to an base64 encoded string
    Attributes
    ----------
    img : object
        Pillow image
    tileformat : str
        Image format to return (Accepted: "jpg" or "png")

    Returns
    -------
    out : str
        base64 encoded image.
    """
    params = {'compress_level': 9}

    if tileformat == 'jpeg':
        img = img.convert('RGB')

    sio = BytesIO()
    img.save(sio, tileformat.upper(), **params)
    sio.seek(0)
    return base64.b64encode(sio.getvalue()).decode()


class TilerError(Exception):
    """Base exception class."""


app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app)

app.config['COMPRESS_MIMETYPES'] = ['text/html', 'text/css', 'text/xml',
                                    'application/json',
                                    'application/javascript',
                                    'image/png',
                                    'image/PNG',
                                    'image/jpg',
                                    'imgae/jpeg',
                                    'image/JPG',
                                    'image/JPEG']
app.config['COMPRESS_LEVEL'] = 9
app.config['COMPRESS_MIN_SIZE'] = 0
Compress(app)


# Remappj
def remap_array(arr):
    """
    Remapping [3, 256,256] to [256,256,3]
    """
    return np.moveaxis(arr, 0, 2)

# Welcome page
@app.route('/')
def hello():
    return "Welcome to, OntheFly API!"


# Generates bounds of Raster data
@app.route('/api/v1/bounds', methods=['GET'])
def bounds():
    print('Going to bounds')
    """Handle bounds requests."""
    url = request.args.get('url', default='', type=str)
    url = requote_uri(url)

    # address = query_args['url']
    info = main.bounds(url)
    return (jsonify(info))


# Generates metadata of raster
@app.route('/api/v1/metadata', methods=['GET'])
def metadata():
    """Handle metadata requests."""
    url = request.args.get('url', default='', type=str)
    url = requote_uri(url)

    # address = query_args['url']
    info = main.metadata(url)
    return (jsonify(info))


# Clipping dataset on the fly
@app.route('/api/v1/clip', methods=['GET'])
def clip():
    """Clips data on the fly."""

    # query_args = APP.current_request.query_params
    # query_args = query_args if isinstance(query_args, dict) else {}

    url = request.args.get('url', default='', type=str)
    url = requote_uri(url)
    print('url: ', url)

    url_shp = request.args.get('shp', default='', type=str)
    url_shp = requote_uri(url_shp)
    print('url_shp: ', url_shp)

    nodata = request.args.get('nodata', default=-9999, type=int)

    # tilesize = request.args.get('tilesize', default=256, type=int)
    # scale = request.args.get('scale', default=2, type=int)

    # tileformat = request.args.get('tileformat', default='tif', type=str)

    if not url:
        raise TilerError("Missing 'url' tif parameter")
    else:
        url = '/vsicurl/' + url

    if not url_shp:
        raise TilerError("Missing 'shp' url parameter")
    else:
        url_shp = '/vsicurl/' + url_shp

    if nodata is not None:
        nodata = int(nodata)


    with fiona.open(url_shp, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    print('number of shapes : %d' % (len(shapes)))

    print('Clipping dataset')
    with rasterio.open(url) as src:
        tile, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta
    
    out_meta.update({"driver": "GTiff",
                 "height": tile.shape[1],
                 "width": tile.shape[2],
                 "transform": out_transform})

    # Saving to output.tif
    print('Saving dataset')
    path_output = './output.tif'
    with rasterio.open(path_output, "w", **out_meta) as dest:
        dest.write(tile)
    
    print('dataset saved')
    return send_file(path_output, as_attachment=True)


@app.route('/api/v1/favicon.ico', methods=['GET'])
def favicon():
    """Favicon."""
    output = {}
    output['status'] = '205'  # Not OK
    output['type'] = 'text/plain'
    output['data'] = ''
    return (json.dumps(output))


if __name__ == '__main__':
    app.run(debug=True, threaded=True)


"""
Status Codes :

200 OK
    Standard response for successful HTTP requests. The actual response will depend on the request method used. In a GET request, the response will contain an entity corresponding to the requested resource. In a POST request, the response will contain an entity describing or containing the result of the action.[9]
201 Created
    The request has been fulfilled, resulting in the creation of a new resource.[10]
202 Accepted
    The request has been accepted for processing, but the processing has not been completed. The request might or might not be eventually acted upon, and may be disallowed when processing occurs.[11]
203 Non-Authoritative Information (since HTTP/1.1)
    The server is a transforming proxy (e.g. a Web accelerator) that received a 200 OK from its origin, but is returning a modified version of the origin's response.[12][13]
204 No Content
    The server successfully processed the request and is not returning any content.[14]
205 Reset Content
    The server successfully processed the request, but is not returning any content. Unlike a 204 response, this response requires that the requester reset the document view.[15]
206 Partial Content (RFC 7233)
    The server is delivering only part of the resource (byte serving) due to a range header sent by the client. The range header is used by HTTP clients to enable resuming of interrupted downloads, or split a download into multiple simultaneous streams.[16]
207 Multi-Status (WebDAV; RFC 4918)
    The message body that follows is by default an XML message and can contain a number of separate response codes, depending on how many sub-requests were made.[17]
208 Already Reported (WebDAV; RFC 5842)
    The members of a DAV binding have already been enumerated in a preceding part of the (multistatus) response, and are not being included again.
226 IM Used (RFC 3229)
    The server has fulfilled a request for the resource, and the response is a representation of the result of one or more instance-manipulations applied to the current instance.
    """
