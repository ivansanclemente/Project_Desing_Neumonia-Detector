# Prueba Unitaria Funcion Cargue Imagen

from array import array
from asyncio.windows_events import NULL
from pickle import TRUE
from inference import infe
import unittest
import os.path
import numpy as np
#import pytest


class TestApp(unittest.TestCase):

    
    def test_1(self):
     
        print("Cargue de Imagen")
        path = os.path.abspath('./images/NORMAL2-IM-0826-0001.jpeg')
        self.img2, self.img2show =infe.read_jpg_file(path)
        self.assertFalse(NULL)
        print(self.img2)

    
        
if __name__ == '__main__':
    unittest.main()



