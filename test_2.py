# Prueba unitaria Funcion Preprocesamiento de Imagen

from array import array
from pickle import TRUE
from inference import infe
import unittest
import os.path
import numpy as np
#import pytest


class TestApp(unittest.TestCase):

    
    def test_2(self):
     
        print("Preprocesamiento Imagen")
        path = os.path.abspath('./images/NORMAL2-IM-0826-0001.jpeg')
        img2, img2show=infe.read_jpg_file(path)
        self.salida=infe.preprocess(img2)
        self.assertTrue(object)
        print(self.salida)     
        
        
if __name__ == '__main__':
    unittest.main()



