# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:21:08 2017

@author: Leevi Annala
"""


import unittest
import numpy as np
import xarray as xr
import holoviews as hv
from visacc import VisualisorAccessor
import sys
# sys.path.append('C:/Users/lealanna/MaskAccessor/')
import pandas as pd

class test(unittest.TestCase):
    def test_init(self):
        cube = xr.open_rasterio('C:/Users/lealanna/spectral_selector/' +
                                'tests/testdata/cube2.dat')
        arr = np.random.rand(3,3,3)
        cube2 = xr.DataArray(arr,
                             dims=['band', 'y', 'x'],
                             coords={'band':[1,2,3], 'y':[1,2,3], 'x':[1,2,3]})
        self.assertTrue(hasattr(cube, 'visualize'))
        self.assertTrue(hasattr(cube2, 'visualize'))
        VisualisorAccessor(cube)
        VisualisorAccessor(cube2)
        self.assertTrue(hasattr(cube, 'visualize'))
        self.assertTrue(hasattr(cube2, 'visualize'))

    def test_basic_2d(self):
        arr = np.random.rand(3,3)

        cube2 = xr.DataArray(arr,
                             dims=['y', 'x'],
                             coords={'y': [1, 2, 3], 'x': [1, 2, 3]})
        cube1 = xr.DataArray(arr,
                             dims=['a', 'b'],
                             coords={'a': [1, 2, 3], 'b': [1, 2, 3]})

        assert isinstance(cube2, xr.DataArray)
        layout_1 = cube1.visualize.basic_2d()
        assert isinstance(cube1, xr.DataArray)
        layout_2 = cube2.visualize.basic_2d()
        self.assertTrue(isinstance(layout_1, hv.Image))
        assert isinstance(layout_1, hv.Image)
        dims_actual_1 = layout_1.dimensions()
        dims_actual_2 = layout_2.dimensions()
        dims_predicted_1 = [hv.Dimension('b'),
                            hv.Dimension('a'),
                            hv.Dimension('Value')]

        dims_predicted_2 = [hv.Dimension('x'),
                            hv.Dimension('y'),
                            hv.Dimension('Value')]
        self.assertEqual(dims_actual_1,dims_predicted_1)
        self.assertEqual(dims_actual_2,dims_predicted_2)

    def test_basic(self):
        arr = np.random.rand(3, 3, 3)
        cube = xr.open_rasterio('C:/Users/lealanna/spectral_selector/' +
                                'tests/testdata/cube2.dat')
        cube2 = xr.DataArray(arr,
                             dims=['y', 'x', 'z'],
                             coords={'y': [1, 2, 3],
                                     'x': [1, 2, 3],
                                     'z': [1, 2, 3]})
        cube1 = xr.DataArray(arr,
                             dims=['a', 'b', 'c'],
                             coords={'a': [1, 2, 3],
                                     'b': [1, 2, 3],
                                     'c': [1, 2, 3]
                                     })
        cube3 = xr.DataArray(arr,
                             dims=['Value', 'b', 'c'],
                             coords={'Value': [1, 2, 3],
                                     'b': [1, 2, 3],
                                     'c': [1, 2, 3]
                                     })
        assert isinstance(cube, xr.DataArray)
        assert isinstance(cube1, xr.DataArray)
        assert isinstance(cube2, xr.DataArray)
        assert isinstance(cube3, xr.DataArray)
        layouts = [cube.visualize.basic(sliders=['band']),
                   cube1.visualize.basic(sliders=['a']),
                   cube2.visualize.basic(sliders=['y']),
                   cube1.visualize.basic(sliders=['b']),
                   cube2.visualize.basic(sliders=['z']),
                   cube2.visualize.basic(sliders=['x']),
                   cube1.visualize.basic(sliders=['c'])]
        with self.assertRaises(ValueError):
            cube3.visualize.basic(sliders=['Value'])
        with self.assertRaises(ValueError):
            cube3.visualize.basic(sliders=['b'])
        with self.assertRaises(ValueError):
            cube3.visualize.basic(sliders=['c'])
        for layout in layouts:
            self.assertTrue(isinstance(layout, hv.AdjointLayout))
        with self.assertRaises(ValueError):
            cube2.visualize.basic(sliders=['x', 'y'])

        layouts_dims = []
        for name in ['band', 'a', 'y', 'b', 'z', 'x', 'c']:
            layouts_dims.append([hv.Dimension('AdjointLayout'),
                                 hv.Dimension(name)])

        for i, layout in enumerate(layouts):
            self.assertEqual(layouts_dims[i], layout.dimensions())

    def test_show_mask(self):
        arr = np.random.rand(3,3,3)
        cube1 = xr.DataArray(arr,
                             dims=['a', 'b', 'c'],
                             coords={'a': [1, 2, 3],
                                     'b': [1, 2, 3],
                                     'c': [1, 2, 3]
                                     })


        arr2 = np.random.rand(4,3,2)
        cube2 = xr.DataArray(arr2,
                             dims=['a', 'b', 'c'],
                             coords={'a': [1, 2, 3, 4],
                                     'b': [1, 2, 3],
                                     'c': [1, 2]
                             })
        cube2.M.refresh(dims=['b','c'])
        layout = cube1.visualize.show_mask()
        layout2 = cube2.visualize.show_mask()
        self.assertTrue(isinstance(layout, hv.Image))
        self.assertTrue(isinstance(layout2, hv.Image))
        dims = [hv.Dimension('c'), hv.Dimension('b'), hv.Dimension('Value')]
        dims2 = [hv.Dimension('c'), hv.Dimension('b'), hv.Dimension('Value')]
        self.assertEqual(layout.dimensions(), dims)
        self.assertEqual(layout2.dimensions(), dims2)
        cube2.M.refresh(dims=['c', 'b'])
        layout2 = cube2.visualize.show_mask()
        self.assertTrue(isinstance(layout2, hv.Image))
        self.assertEqual(layout2.dimensions(), dims2)
        cube2.M.refresh(dims=['a', 'c'])
        layout2 = cube2.visualize.show_mask()
        dims2 = [hv.Dimension('c'), hv.Dimension('a'), hv.Dimension('Value')]
        self.assertTrue(isinstance(layout2, hv.Image))
        self.assertEqual(layout2.dimensions(), dims2)
        cube2.M.refresh(dims=['b', 'a'])
        layout2 = cube2.visualize.show_mask()
        dims2 = [hv.Dimension('b'), hv.Dimension('a'), hv.Dimension('Value')]
        self.assertTrue(isinstance(layout2, hv.Image))
        self.assertEqual(layout2.dimensions(), dims2)

    def test__record_taps(self):
        arr2 = np.random.rand(4,3,2)
        cube = xr.DataArray(arr2,
                             dims=['a', 'b', 'c'],
                             coords={'a': [1, 2, 3, 4],
                                     'b': [1, 2, 3],
                                     'c': [1, 2]})
        cube.M.selected_zeros() # Mask width (first coord) is 3 and height 2
        exp_res = pd.DataFrame(data=[],columns=['b','c'])
        res = cube.visualize._record_taps(-1, -1)
        self.assertTrue((exp_res == res).all)

        res = cube.visualize._record_taps(0, -1)
        self.assertTrue((exp_res == res).all)

        res = cube.visualize._record_taps(-1, 0)
        self.assertTrue((exp_res == res).all)

        res = cube.visualize._record_taps(-1, 1)
        self.assertTrue((exp_res == res).all)

        res = cube.visualize._record_taps(-1, 2)
        self.assertTrue((exp_res == res).all)

        res = cube.visualize._record_taps(-1, 3)
        self.assertTrue((exp_res == res).all)

        res = cube.visualize._record_taps(1, -1)
        self.assertTrue((exp_res == res).all)

        res = cube.visualize._record_taps(2, -1)
        self.assertTrue((exp_res == res).all)

        res = cube.visualize._record_taps(3, -3)
        self.assertTrue((exp_res == res).all)

        res = cube.visualize._record_taps(-1, 0)
        self.assertTrue((exp_res == res).all)

        exp_res = pd.DataFrame(data=[[0,0]], columns=['b', 'c'])
        res = cube.visualize._record_taps(0, 0)
        self.assertTrue((exp_res == res).all)

        exp_res = exp_res = pd.DataFrame(np.array([[0, 0], [1, 1]]), columns=['b', 'c'])
        res = cube.visualize._record_taps(0.51, 0.6)
        self.assertTrue((exp_res == res).all)

        res = cube.visualize._record_taps(3, 1)
        self.assertTrue((exp_res == res).all)

        exp_res = pd.DataFrame(data=np.array([[0, 0], [1, 1], [2, 1]]),
                               columns=['b','c'])
        res = cube.visualize._record_taps(1.45, 1.89)
        self.assertTrue((exp_res == res).all)

        exp_res = pd.DataFrame(data=np.array([[0, 0], [0, 1], [1, 1], [2, 1]]),
                               columns=['b', 'c'])
        res = cube.visualize._record_taps(1.45, 0.45)
        self.assertTrue((exp_res == res).all)

        exp_res = pd.DataFrame(data=np.array([[0, 0], [0, 1], [1, 1], [2, 1]]),
                               columns=['b', 'c'])
        res = cube.visualize._record_taps(0.45, 2.6)
        self.assertTrue((exp_res == res).all)

        exp_res = pd.DataFrame(data=np.array([[0, 0], [0, 1], [1, 1], [2, 1]]),
                               columns=['b', 'c'])
        res = cube.visualize._record_taps(2.45, 2.0)
        self.assertTrue((exp_res == res).all)

        exp_res = pd.DataFrame(data=np.array([[0, 0], [0, 1], [1, 1], [2, 1]]),
                               columns=['b', 'c'])
        res = cube.visualize._record_taps(2.0, 2.45)
        self.assertTrue((exp_res == res).all)

        exp_res = pd.DataFrame(data=np.array([[0, 0], [0, 1], [1, 0],
                                              [1, 1], [2, 0], [2, 1]]),
                               columns=['b', 'c'])
        cube.visualize._record_taps(0.25,1.54)
        res = cube.visualize._record_taps(0.49, 1.45)
        self.assertTrue((exp_res == res).all)


        cube.M.refresh(dims=['a', 'b'])
        cube.M.selected_zeros()  # Mask width (first coord) is 4 and height 3
        exp_res = pd.DataFrame(data=np.array([[0, 0], # 3
                                              [0, 2], # 1
                                              [1, 2], # 4
                                              [2, 1], # 6
                                              [3, 0]]), # 5
                               columns=['a','b'])
        cube.visualize._record_taps(1.6, -0.49) # 1
        cube.visualize._record_taps(1.6, -0.51) # 2
        cube.visualize._record_taps(0.49,-0.49) # 3
        cube.visualize._record_taps(2, 1) # 4
        cube.visualize._record_taps(0.000058, 3.33) # 5
        cube.visualize._record_taps(1.49, 2.21) # 6
        res = cube.visualize._record_taps(1.6, 4) # 7
        self.assertTrue((exp_res == res).all)
        self.assertTrue((cube==cube.visualize._obj).all)
        self.assertTrue(cube.M == cube.visualize._obj.M)
        self.assertTrue((cube.M.mask == cube.visualize._obj.M.mask).all)
        self.assertTrue(cube.M.dims == cube.visualize._obj.M.dims)

    def test_point_chooser(self):
        arr2 = np.random.rand(4, 3, 2)
        cube = xr.DataArray(arr2,
                            dims=['a', 'b', 'c'],
                            coords={'a': [1, 2, 3, 4],
                                    'b': [1, 2, 3],
                                    'c': [1, 2]})
        layout = cube.visualize.point_chooser()
        self.assertTrue(isinstance(layout, hv.DynamicMap))

        layout = cube.visualize.point_chooser(array=None,
                                              initialize_mask=True,
                                              initialize_value=0)

        self.assertTrue(isinstance(layout, hv.DynamicMap))

        layout = cube.visualize.point_chooser(array=[],
                                              initialize_mask=True,
                                              initialize_value=100)

        self.assertTrue(isinstance(layout, hv.DynamicMap))
        exp_res = np.array([[100,100],[100,100],[100,100]])
        self.assertTrue((exp_res == cube.M.mask).all)
        with self.assertRaises(TypeError):
            cube.visualize.point_chooser(array=[[1,2,3][4,5,6]],
                                         initialize_mask=False,
                                         initialize_value=100)

        layout = cube.visualize.point_chooser(array=[[0,1],[1,1]],
                                              initialize_mask=False,
                                              initialize_value=100)
        self.assertTrue(isinstance(layout, hv.DynamicMap))
        exp_res = np.array([[100, 1], [100, 1], [100, 100]])
        self.assertTrue((exp_res == cube.M.mask).all)

    def test_is_in(self):
        arr2 = np.random.rand(4, 3, 2)
        cube = xr.DataArray(arr2,
                            dims=['a', 'b', 'c'],
                            coords={'a': [1, 2, 3, 4],
                                    'b': [1, 2, 3],
                                    'c': [1, 2]})
        third_dim_data = [1, 2, 3, 4]
        bounds = [0, 0, 5, 5]
        spectre = [1, 1, 2, 1]
        self.assertTrue(cube.visualize._is_in(spectre, third_dim_data, bounds))

        bounds = [2.1, 1, 2.9, 2]
        self.assertTrue(cube.visualize._is_in(spectre, third_dim_data, bounds))

        spectre = [6, 7, 6, 7]

        self.assertFalse(
            cube.visualize._is_in(spectre, third_dim_data, bounds))

        bounds = [2.1, 5, 2.9, 7]
        self.assertTrue(cube.visualize._is_in(spectre, third_dim_data, bounds))

        spectre = [6, 5, 4, 7]

        self.assertFalse(
            cube.visualize._is_in(spectre, third_dim_data, bounds))

        third = [2, 3, 4, 5]
        spec = [2, 3.5, 2.5, 4.7]
        bounds = [3.4, 2.8, 3.6, 3.5]
        self.assertTrue(cube.visualize._is_in(spec, third, bounds))
        third = [2, 4, 5, 3]
        spec = [2, 2.5, 4.7, 3.5]
        self.assertFalse(cube.visualize._is_in(spec, third, bounds))
        bounds = [1, 1, 6, 5]
        self.assertTrue(cube.visualize._is_in(spec, third, bounds))
        bounds = [2.5, 1.5, 3.5, 2.5]
        self.assertTrue(cube.visualize._is_in(spec, third, bounds))

    def test_box_chooser(self):
        arr2 = np.random.rand(4, 3, 2)
        cube = xr.DataArray(arr2,
                            dims=['a', 'b', 'c'],
                            coords={'a': [1, 2, 3, 4],
                                    'b': [1, 2, 3],
                                    'c': [1, 2]})
        layout = cube.visualize.box_chooser()
        self.assertTrue(isinstance(layout, hv.Layout))

        layout = cube.visualize.box_chooser(array=None,
                                              initialize_mask=True,
                                              initialize_value=0)

        self.assertTrue(isinstance(layout, hv.Layout))

        layout = cube.visualize.box_chooser(array=[],
                                              initialize_mask=True,
                                              initialize_value=100)

        self.assertTrue(isinstance(layout, hv.Layout))
        exp_res = np.array([[100, 100], [100, 100], [100, 100]])
        self.assertTrue((exp_res == cube.M.mask).all)
        with self.assertRaises(TypeError):
            cube.visualize.box_chooser(array=[[1, 2, 3][4, 5, 6]],
                                         initialize_mask=False,
                                         initialize_value=100)

        layout = cube.visualize.box_chooser(array=[[0, 1], [1, 1]],
                                              initialize_mask=False,
                                              initialize_value=100)
        self.assertTrue(isinstance(layout, hv.Layout))
        exp_res = np.array([[100, 1], [100, 1], [100, 100]])
        self.assertTrue((exp_res == cube.M.mask).all)

    def test_spectre_chooser(self):
        arr2 = np.random.rand(4, 3, 2)
        cube = xr.DataArray(arr2,
                            dims=['a', 'b', 'c'],
                            coords={'a': [1, 2, 3, 4],
                                    'b': [1, 2, 3],
                                    'c': [1, 2]})
        layout = cube.visualize.spectre_chooser()
        self.assertTrue(isinstance(layout, hv.DynamicMap))

        layout = cube.visualize.spectre_chooser(array=None,
                                              initialize_mask=True,
                                              initialize_value=0)

        self.assertTrue(isinstance(layout, hv.DynamicMap))

        layout = cube.visualize.spectre_chooser(array=[],
                                              initialize_mask=True,
                                              initialize_value=100)

        self.assertTrue(isinstance(layout, hv.DynamicMap))
        exp_res = np.array([[100, 100], [100, 100], [100, 100]])
        self.assertTrue((exp_res == cube.M.mask).all)
        with self.assertRaises(TypeError):
            cube.visualize.spectre_chooser(array=[[1, 2, 3][4, 5, 6]],
                                         initialize_mask=False,
                                         initialize_value=100)

        layout = cube.visualize.spectre_chooser(array=[[0, 1], [1, 1]],
                                              initialize_mask=False,
                                              initialize_value=100)
        self.assertTrue(isinstance(layout, hv.DynamicMap))
        exp_res = np.array([[100, 1], [100, 1], [100, 100]])
        self.assertTrue((exp_res == cube.M.mask).all)

    def test_appropriate_amount_of_graphs(self):
        arr2 = np.random.rand(4, 3, 2)
        cube = xr.DataArray(arr2,
                            dims=['a', 'b', 'c'],
                            coords={'a': [1, 2, 3, 4],
                                    'b': [1, 2, 3],
                                    'c': [1, 2]})
        res = cube.visualize._appropriate_amount_of_graphs()
        self.assertTrue(len(res) < 3000)

        arr2 = np.random.rand(4000, 3, 2)
        cube = xr.DataArray(arr2,
                            dims=['a', 'b', 'c'],
                            coords={'a': range(4000),
                                    'b': [1, 2, 3],
                                    'c': [1, 2]})
        res = cube.visualize._appropriate_amount_of_graphs()
        self.assertTrue(len(res) < 3000)

        arr2 = np.random.rand(1000, 1000, 2)
        cube = xr.DataArray(arr2,
                            dims=['a', 'b', 'c'],
                            coords={'a': range(1000),
                                    'b': range(1000),
                                    'c': [1, 2]})
        res = cube.visualize._appropriate_amount_of_graphs()
        self.assertTrue(len(res) < 3000)

        arr2 = np.random.rand(40, 1000, 1000)
        cube = xr.DataArray(arr2,
                            dims=['a', 'b', 'c'],
                            coords={'a': range(40),
                                    'b': range(1000),
                                    'c': range(1000)})
        res = cube.visualize._appropriate_amount_of_graphs()
        self.assertTrue(len(res) < 3000)

        arr2 = np.random.rand(1000, 10, 1000)
        cube = xr.DataArray(arr2,
                            dims=['a', 'b', 'c'],
                            coords={'a': range(1000),
                                    'b': range(10),
                                    'c': range(1000)})
        res = cube.visualize._appropriate_amount_of_graphs()
        self.assertTrue(len(res) < 3000)

    def test_choosing_spectre(self):
        '''
        Mostly tested in test_is_in and test_appropriate_amount_of_graphs
        '''
        arr2 = np.random.rand(4, 3, 2)
        cube = xr.DataArray(arr2,
                            dims=['a', 'b', 'c'],
                            coords={'a': [1, 2, 3, 4],
                                    'b': [1, 2, 3],
                                    'c': [1, 2]})
        bounds = [0, 1, 2, 3]
        exp_ret = []
        for i in [1,2,3]:
            for j in [1, 2]:
                spectre = cube.sel(b=i, c=j).data
                if cube.visualize._is_in(spectre=spectre,
                                         third_dim_data=[1,2,3,4],
                                         bounds=bounds):
                    exp_ret.append(spectre)
        exp_ret = np.array(exp_ret)
        ret = cube.visualize._new_choosing_spectre(bounds=bounds)
        self.assertTrue((exp_ret == ret).all)

    def test_record_selections(self):
        arr2 = np.random.rand(4, 3, 2)
        cube = xr.DataArray(arr2,
                            dims=['a', 'b', 'c'],
                            coords={'a': [1, 2, 3, 4],
                                    'b': [1, 2, 3],
                                    'c': [1, 2]})
        cube.M.selected_zeros()
        bounds = [0, 0, 5, 5]
        points = hv.Points(np.array([[0,0],[1,1],[1,0]]), kdims=['c','b'])
        res = cube.visualize._record_selections(bounds=bounds, points=points)
        exp_res = pd.DataFrame(columns=['b', 'c'], data=[[0,0], [0,1], [1,1]])
        self.assertTrue((exp_res==res).all)

        points = hv.Points(np.array([]), kdims=['c', 'b'])
        res = cube.visualize._record_selections(bounds=bounds, points=points)
        self.assertTrue((exp_res == res).all)

        bounds = (0,0,10,10)
        cube.M.selected_zeros()
        points = hv.Points(np.array([]), kdims=['c', 'b'])
        exp_res = pd.DataFrame(columns=['b', 'c'],
                               data=[])
        res = cube.visualize._record_selections(bounds=bounds, points=points)
        self.assertTrue((exp_res == res).all)

    def test_make_dataset_opts(self):
        arr = np.random.rand(4, 3, 2)
        cube1 = xr.DataArray(arr,
                             dims=['a', 'b', 'c'],
                             coords={'a': [1, 2, 3, 4],
                                     'b': [1, 2, 3],
                                     'c': [1, 2]
                                     })
        exp_res = {'data':(range(0,2), range(0,3), [1,2,3,4], cube1),
                   'kdims': ['c', 'b', 'a'],
                   'vdims':['Value']}
        res = cube1.visualize._make_dataset_opts()
        self.assertTrue(res == exp_res)
        cube1.M.refresh(dims=['b', 'a'])
        exp_res = {'data': ([1, 2], range(0, 3), range(0, 4), cube1),
                   'kdims': ['c', 'b', 'a'],
                   'vdims': ['Value']}
        res = cube1.visualize._make_dataset_opts()
        self.assertTrue(res == exp_res)

    def test_histogram(self):
        data = [[[1,2,2,3],[1,3,2,5]],
                [[3,2,2,2],[2,6,4,5]],
                [[5,78,32,5],[1,32,4,7]]]
        cube = xr.DataArray(data,
                            dims=['y', 'x', 'z'],
                            coords={'y':[1, 2, 3],
                                    'x':[1, 2],
                                    'z':[1, 2, 3, 4]})
        layout, hist, bin_edges = cube.visualize.histogram(band_dim='y')
        self.assertTrue(len(bin_edges == 3))
        len_1 = bin_edges[1] - bin_edges[0]
        len_2 = bin_edges[2] - bin_edges[1]
        exp_res = np.array([[8 / len_1, 0 / len_2],
                            [8 / len_1, 0 / len_2],
                            [7 / len_1, 1 / len_2]])
        self.assertTrue((hist == exp_res).all)

        layout, hist, bin_edges = cube.visualize.histogram(band_dim='x')
        self.assertTrue(len(bin_edges == 4))
        len_1 = bin_edges[1] - bin_edges[0]
        len_2 = bin_edges[2] - bin_edges[1]
        len_3 = bin_edges[3] - bin_edges[2]
        exp_res = np.array([[12 / len_1, 1 / len_2, 1 / len_3],
                            [13 / len_1, 1 / len_2, 0 / len_3]])
        self.assertTrue((hist == exp_res).all)

        layout, hist, bin_edges = cube.visualize.histogram(band_dim='z')
        self.assertTrue(len(bin_edges == 2))
        len_1 = bin_edges[1] - bin_edges[0]
        len_2 = bin_edges[2] - bin_edges[1]
        exp_res = np.array([[6 / len_1, 0 / len_2],
                            [5 / len_1, 1 / len_2],
                            [6 / len_1, 0 / len_2],
                            [6 / len_1, 0 / len_2]])
        self.assertTrue((hist == exp_res).all)

        bin_edges = [0,1,2,10,100]
        layout, hist, bin_edges_2 = cube.visualize.histogram(band_dim='y',
                                                           bin_edges=bin_edges,
                                                           visualize = False)
        self.assertTrue(layout == None)
        self.assertTrue(bin_edges == bin_edges_2)
        len_1 = 1
        len_2 = 1
        len_3 = 8
        len_4 = 90
        exp_res = np.array([[0 / len_1, 2 / len_2, 6 / len_3, 0 / len_4],
                            [0 / len_1, 0 / len_2, 8 / len_3, 0 / len_4],
                            [0 / len_1, 1 / len_2, 4 / len_3, 3 / len_4]])
        self.assertTrue((exp_res == hist).all)



if __name__ == '__main__':
    unittest.main()