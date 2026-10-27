"""Reject adversarial NPY headers before allocation, including tiny ZIP bombs."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile
import numpy as np
from ghosttrace.config import EngineError
from ghosttrace.storage.artifacts import read_activation


def payload(shape=(1, 1, 1), descr='<f4', data=b''):
    stream=io.BytesIO()
    np.lib.format.write_array_header_1_0(stream,{'shape':shape,'fortran_order':False,'descr':descr})
    stream.write(data)
    return stream.getvalue()


class ActivationHeaderContracts(unittest.TestCase):
    def test_invalid_headers_rejected_before_numpy_load(self):
        cases=[payload(shape=(1000000,1000000,1000000)),payload(shape=(0,1,1)),payload(shape=(2,3)),payload(descr='|O'),payload(descr='<f8'),payload(shape=(1,1,1),data=b''),payload(shape=(1,1,1),data=b'\0'*8)]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'bad.npz'
            for content in cases:
                with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as archive:archive.writestr('activations.npy',content)
                with self.subTest(header_bytes=len(content)), patch('numpy.load',side_effect=AssertionError('Allocation was reached before validation')):
                    with self.assertRaises(EngineError) as error:read_activation(path)
                    self.assertEqual(error.exception.code,'ARTIFACT_CORRUPT')

    def test_duplicate_or_unexpected_npz_members_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'bad.npz'
            for names in [('other.npy',),('activations.npy','other.npy'),('activations.npy','activations.npy')]:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',UserWarning)
                    with zipfile.ZipFile(path,'w') as archive:
                        for name in names:archive.writestr(name,payload(data=b'\0'*4))
                with self.subTest(names=names), patch('numpy.load',side_effect=AssertionError('Unexpected payload loaded')):
                    with self.assertRaises(EngineError):read_activation(path)

    def test_valid_payload_reads_without_pickle(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'valid.npz'
            with zipfile.ZipFile(path,'w') as archive:archive.writestr('activations.npy',payload(data=np.float32(1.25).tobytes()))
            array=read_activation(path)
            self.assertEqual(array.shape,(1,1,1));self.assertEqual(array.dtype,np.float32)
            self.assertEqual(array[0,0,0],1.25)
