# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:09:42 2019

A Package for Reading/Writing IDX Files
---------------------------------------
IDX files are a convenient way to store numeric tensors (including vectors 
and matrices). The basic file format is:
    
    Type          : Description
    -----------------------------------
    32-bit integer: magic number
    32-bit integer: size in dimension 0
    32-bit integer: size in dimension 1
    ...
    32-bit integer: size in dimension n
    data

The magic number contains 4 bytes. The first two bytes are always 0 (which 
allows you to check the byte order of the file: little-endian or bit-endian).
The next byte encodes the data type:
    
    Standard IDX Formats
    0x08: unsigned byte
    0x09: signed byte
    0x0B: short (2 bytes)
    0x0C: int (4 bytes)
    0x0D: float (4 bytes)
    0x0E: double (8 bytes)
    
    Non-Standard IDX Formats Supported by this Package
    0x0F: (6)/16-bit float (2 bytes)
    0x10: long (8 bytes)

And the last byte of the magic number stores the number of dimensions. 
Finally, the data is stored like a c-array, where the last index 
changes the fastest.


The main methods in this package are:
    read(file)              - Read an IDX File
    write(array, filename)  - Write an Array to an IDX File

There are 2 additional helper-methods:
    checkFilename(filename) - Check the Validity of a Filename
    Open(filename)          - Open a File (Returns a byte string)

For help with any of these methods, please see their respective documentation.

Please note that this package is dependent of the following packages:
    os
    gzip
    struct
    numpy

@author: Violet Saathoff
"""

#Import Libraries
import os
import gzip
import struct
import numpy as np

#A Dictionary of Data Types (dtype:(idx code, struct code, size))
dtypes = {np.uint8:(8, 'B', 1),
          np.int8:(9, 'b', 1),
          np.int16:(10, 'h', 2),
          np.int32:(11, 'i', 4),
          np.float32:(12, 'f', 4),
          np.float64:(13, 'd', 8),
          np.float16:(14, 'e', 2), #Not A Standard IDX Data Format
          np.int64:(15, 'q', 8),   #Not A Standard IDX Data Format
          int:(11, 'i', 4),        #Overloading for Convenience
          float:(13, 'd', 8)}      #Overloading for Convenience

#Check a Filename
def checkFilename(filename:str, suffix:str = '.idx') -> str:
    """Check and Format a Filename"""
    #Check the Type of the Suffix
    if type(suffix) != str:
        raise TypeError("'suffix' must be a stirng. Current type: " + str(type(suffix)))
    
    #Check the Value of Suffix
    suffix = suffix.lower()
    if suffix not in {'.idx', '.gz'}:
        #Try to Fix the Suffix
        if suffix in {'idx', 'gz'}:
            #Fix the Suffix
            suffix = '.' + suffix
        else:
            #Raise a ValueError
            raise ValueError('\n'.join(["Invalid Suffix: " + suffix,
                                        'Valid Suffixes:',
                                        "    '.idx' : Uncompressed",
                                        "    '.gz' : Compressed"]))
    
    #Check the Type of the Filename
    if type(filename) != str:
        raise TypeError("'filename' must be a string. Current type: " + str(type(filename)))
    
    #Strip the Suffix
    if filename[-len(suffix):] == suffix:
        #Remove the Suffix
        filename = filename[-len(suffix):]
    
    #Remove Leading/Trailing Whitespace
    filename = filename.strip('.')
    
    #Make a Copy With Some Valid Non-AlphaNumeric Charactars Removed
    clean = filename
    chars = [' ', '_', '-', '!']
    for char in chars:
        clean = clean.replace(char, '')
    
    #Check the Remaining Characters
    if not clean.isalnum():
        #Find the Invalid Character
        for char in clean:
            if not char.isalnum():
                #Raise an Exception
                raise ValueError("Invalid Character '%s' in 'filename'" % char)
    
    #Add the Suffix Back On and Return
    return filename + suffix

#Open a File
def Open(filename:str) -> bytes:
    """Open an IDX File
    
    Parameters
    ----------
    filename : str
        The name of the file you wish to open. The file type must either 
        be '.idx' for uncompressed files, or '.gz' for compressed files.

    Returns
    -------
    file : bytes
        The IDX file as a byte string.

    """
    
    #Check the Filename
    if type(filename) != str:
        raise TypeError("'filename' must be a string. Current type: " + str(type(filename)))
    
    #Get the Suffix
    suffix = filename.split('.')[-1]
    
    #Check the Suffix
    if suffix not in {'idx', 'gz'}:
        raise ValueError("The suffix must either be '.idx', or '.gz'. Current suffix: ." + suffix)
    
    #Read the File
    if suffix == 'gz':
        #Open the File
        with gzip.open(filename) as f:
            file = f.read()
        
        #Close the Buffered Reader
        f.close()
    else:
        #Open the File
        f = open(filename, 'rb')
        
        #Read the File
        file = f.read()
        
        #Close the File
        f.close()
        
    #Return the File
    return file

#Read an IDX File
def read(file) -> np.ndarray:
    """Read an IDX File
    
    Parameters
    ----------
    file : str, bytes
        Either the name of the file you wish to read, or an already-opened 
        byte string.
    
    Returns
    -------
    np.ndarray
        An array containing the parsed data from the file.
    
    Notes
    -----
    When reading the data, it will first check if the data is 
    little-endian or bit-endian (using the magic number), and 
    then automatically select the appropriate byte order.
    
    """
    
    #Check the File
    if type(file) == str:
        #If the file is a String, Try to Interpret it as a Filename
        file = Open(file)
    elif type(file) != bytes:
        #Make Sure the File is a bytes Object
        raise TypeError('file must be a byte string (a bytes object) or a valid filename')
    
    #Define a Function For Reading a Single Data Element
    def parse(i, code, step):
        return struct.unpack(code, file[i:i + step])[0]
    
    #Try Both Byte Orders
    for byteorder in ['<i','>i']:
        #Compute the Magic Number
        magic = parse(0, byteorder, 4)
        
        #Check if the Magic Number is Valid
        if magic < 4096:
            break
    
    #Check that the First 2 Bytes are 0
    if magic >= 4096:
        raise ValueError('Magic Number ' + str(magic) + ' is Too Large')
    
    #Get the Data Type (ubyte, byte, short, int, float, double)
    D = magic >> 8 #D = magic//256
    dtype = list(dtypes.keys())[D - 8]
    D, code, step = dtypes[dtype]
    code = byteorder[0] + code
    
    #Get the Number of Dimensions
    rank = magic % 256
    
    #Get the Size of Each Dimension (The Shape of the Data)
    shape = []
    for i in range(4, 4*(rank + 1), 4):
        shape.append(parse(i, byteorder, 4))
    
    #Read the Rest of the Data Into an Array
    return np.array([parse(i, code, step) for i in range(i + 4, len(file), step)], dtype).reshape(shape)

#Write a File
def write(array:np.ndarray, filename:str, suffix:str = '.idx', byteorder:str = '@') -> None:
    """Write an Array to an IDX File
    
    Parameters
    ----------
    array : np.ndarray
        The array you wish to save.
    filename : str
        The filename you wish to save the array as.
    suffix : str ('.idx', or '.gz'), optional
        The suffix you wish to use when saving the data.
            '.idx' - Save the data without compression
            'gz'   - Save the data with compression
        The default is '.idx'
    byteorder: str ('@', '=', '<', '>'), optional
        The byteorder to use when saving the data.
            @ : Native
            = : Native (Standardized)
            < : Little-Endian
            > : Big-Endian
            ! : Network (Big-Endian)
        The default is '@'
    
    """
    
    #Check the Filename
    filename = checkFilename(filename, suffix)
    suffix = filename.split('.')[-1]
    
    #Check the Byte Order
    if byteorder not in {'@', '=', '<', '>', '!'}:
        raise ValueError('\n'.join(["Unrecognized Byte Order: " + str(byteorder),
                                    'Valid Byteorders:',
                                    "    '@' : Native",
                                    "    '=' : Native (Standardized)",
                                    "    '<' : Little-Endian",
                                    "    '>' : Big-Endian",
                                    "    '!' : Network (Big-Endian)"]))
    
    #Check the Array
    if type(array) not in {list, np.ndarray}:
        raise TypeError('array must be iterable (list/array)')
    
    #Get the Shape of the Array
    shape = np.shape(array)
    
    #Flatten the Array
    array = np.ravel(array)
    
    #Get the Data Type
    dtype = type(array[0])
    if dtype in dtypes:
        #Get the Information About the Data Type
        D, code, size = dtypes[dtype]
        
        #Add the Prefix to the Code
        code = byteorder + code
    else:
        raise TypeError("Unrecognized data type: " + str(dtype))
    
    #Initialize the File with the Magic Number
    file = struct.pack('i', (D << 8) + len(shape))
    
    #Add the Shape of the Array to the File
    for n in shape:
        file += struct.pack('i', n)
    
    #Write the File
    for x in array:
        file += struct.pack(code, x)
    
    #Open a File to Write
    if suffix == 'idx':
        f = open(os.getcwd() + '\\' + filename, 'wb')
    else:
        f = gzip.open(os.getcwd() + '\\' + filename, 'wb')
    
    #Save the Data
    f.write(file)
    f.close()