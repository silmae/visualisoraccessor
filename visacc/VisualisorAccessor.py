# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:40:49 2017

@author: Leevi Annala
"""

import itertools
import xarray as xr
import holoviews as hv
from holoviews import streams
from holoviews.operation import decimate
import pandas as pd
import sys
# sys.path.append('C:/Users/lealanna/MaskAccessor')
from maskacc import MaskAccessor
from .__functions import line, intersection
import numpy as np
import warnings
import math

#@xr.register_dataset_accessor('visualize')
@xr.register_dataarray_accessor('visualize')
class VisualisorAccessor(object):
    '''
    This is xarray add-on for visualisation
    of 3D xarray. This uses holoviews. Works
    together with MaskAccessor

    If you need a visualisation, that is not implemented here, I suggest that
    you take a look at holoviews and its tutorials.
    TODO: FInd out if this works as dataset_accessor
    '''
    def __init__(self, xarray_obj):
        '''
        Initializes the visualisorAccessor
        :param xarray_obj: object that the visualisor is attached to.
        :param kwargs:
        '''
        self._obj = xarray_obj

    def basic_2d(self):
        '''
        Basic 2D visualisation. Returns just a holoviews image, wihout anything
        fancy.
        The visualized object needs to be 2 dimensional.

        :return: Holoviews image
        '''
        return hv.Image(self._obj, vdims=['Value'])

    def basic(self, sliders):
        '''
        Basic visualisation for xarray.
        Usage: if you have a xarray.DataArray named cube with dimensions
        ('x', 'y', 'band'), you can use this as:
        cube.visualize.basic(sliders=['band']) #or
        cube.visualize.basic(sliders=['x']) #or
        cube.visualize.basic(sliders=['y'])
        if you have a xarray.DataArray named cube with dimensions
        ('x','y','z','a'), you can use this for example as:
        cube.visualize.basic(sliders=['y', 'z'])
        and so on.

        If you have 2D data, use basic_2D

        :param sliders: The dimensions you want to use as sliders
        :return: hv.AdjointLayout, with two dimensional images for each slider
        value combination.
        '''
        if not hasattr(sliders, '__len__') or isinstance(sliders, str):
            # Sliders should be a list-like type.
            raise TypeError('sliders should be list-like object.')
        if len(self._obj.dims) < 3:
            raise ValueError('The xarray.DataArray has too few dimensions. ' +
                             'Use basic_2D instead.')
        dimensions_for_ds = list(self._obj.dims) # Making a list for hv.Dataset
        dimensions_for_image = [] # Making a list for hv.Image
        for dimension in dimensions_for_ds:
            if not dimension in sliders:
                dimensions_for_image.append(dimension)
                # dimensions for image
        if len(dimensions_for_ds) - len(sliders) < 2:
            # Too many sliders.
            raise ValueError('You want too many sliders. Can\'t work like ' +
                             'this')
        elif len(dimensions_for_ds) - len(sliders) > 2:
            # Too few sliders.
            raise ValueError('You want too few sliders. Can\'t work like this')
        ds = hv.Dataset(self._obj, kdims=dimensions_for_ds, vdims=['Value'])
        # Making holoviews dataset and returning it as a set of hv.Images.
        return ds.to(hv.Image,
                     dimensions_for_image,
                     'Value',
                     sliders,
                     dynamic=True).hist()

    def show_mask(self):
        '''
        If xarray object has attribute M, shows the mask.
        :return: Mask as a hv.Image. You might want to use magic to see heatmap
        '''
        return hv.Image(self._obj.M.mask_as_xarray(), vdims=['Value'])

    def point_chooser(self, **kwargs):
        '''
        Visual tool for choosing points in the mask.
        :param kwargs: array, if you want to pre select pixels,
                       initialize_mask, True if you want to initialize mask.
                                        True also Overrides array.
                                        False does nothing and overrides
                                        nothing
                       initialize_value, what value you want to use in
                                        initialisation. Used if initialize_mask
                                        is True. Fallback value is 0.
        :return: hv.Object
        '''
        self.__do_mask_changes(**kwargs)
        tap = streams.Tap(rename={'x': 'first_coord', 'y': 'second_coord'},
                          transient=True)
        ds_attrs = self._make_dataset_opts()
        dataset3d = hv.Dataset(**ds_attrs)
        mask_dims = self._obj.M.dims.copy()
        mask_dims.reverse()
        layout = dataset3d.to(hv.Image,
                              mask_dims,
                              'Value',
                              self._obj.M.no_mask_dims) * \
                 decimate(
                     hv.DynamicMap(
                         lambda first_coord, second_coord:
                         hv.Points(self._record_taps(first_coord,
                                                     second_coord),
                                   kdims=mask_dims),
                         streams=[tap]
                     )
                 )
        return layout

    def spectre_chooser(self, **kwargs):
        '''
        Visual tool
        for visualizing selected spectra and narrowing down mask based
        on spectra. It is easy to make a mistake with this, so you should
        keep a backup of your mask. You shouldn't use this on very large sets.
        TODO: This algorithm seems quite unefficient, maybe adding threading
        TODO:        could help.
        :param kwargs: array, if you want to pre select pixels,
                       initialize_mask, True if you want to initialize mask.
                                        True also Overrides array.
                                        False does nothing and overrides
                                        nothing
                       initialize_value, what value you want to use in
                                        initialisation. Used if initialize_mask
                                        is True. Fallback value is 0.
        :return: hv.Object
        '''
        if not len(self._obj.shape) == 3:
            raise ValueError('The spectre_chooser only works for 3 ' +
                             'dimensional objects.')
        self.__do_mask_changes(**kwargs)
        third_dimension = self._obj.M.no_mask_dims[0]
        third_dim_list = self._obj.coords[third_dimension].data
        points = hv.Points(np.array([(np.min(third_dim_list),
                                      np.min(self._obj.data))]))
        box = hv.streams.BoundsXY(
            source=points,
            # bounds is defined as (x_min, y_min, x_max, y_max)
            bounds=(np.min(third_dim_list) - 0.001,
                    np.min(self._obj.data) - 0.001,
                    np.max(third_dim_list) + 0.001,
                    np.max(self._obj.data) + 0.01)
        )

        bounds = hv.DynamicMap(lambda bounds: hv.Bounds(bounds), streams=[box])
        layout = points *\
                 hv.DynamicMap(
                     lambda bounds: hv.Overlay(
                         [hv.Curve((third_dim_list,zs),
                                   kdims=self._obj.M.no_mask_dims,
                                   vdims=['Value'])
                          for zs in self._new_choosing_spectre(bounds)]
                     ),
                     streams=[box]) * \
                 bounds
        return layout

    def box_chooser(self, **kwargs):
        '''
        Visual tool for choosing rectangle shaped blocks in mask.
        :param kwargs: array, if you want to pre select pixels,
                       initialize_mask, True if you want to initialize mask.
                                        True also Overrides array.
                                        False does nothing and overrides
                                        nothing
                       initialize_value, what value you want to use in
                                        initialisation. Used if initialize_mask
                                        is True. Fallback value is 0.
        :return: hv.Object
        '''
        # cube.test_for_not_none()
        wid = len(self._obj.coords[self._obj.M.dims[1]])
        hei = len(self._obj.coords[self._obj.M.dims[0]])
        # dep = len(cube.band)

        self.__do_mask_changes(**kwargs)
        all_pixels = np.array(list(itertools.product(range(wid), range(hei))))
        mask_dims = self._obj.M.dims.copy()
        mask_dims.reverse()
        points = hv.Points(all_pixels, kdims=mask_dims)
        # print(points)
        ds_attrs = self._make_dataset_opts()
        dataset3d = hv.Dataset(**ds_attrs)
        box = hv.streams.BoundsXY(source=points, bounds=(0, 0, 0, 0))
        bounds = hv.DynamicMap(lambda bounds: hv.Bounds(bounds), streams=[box])
        third_dim_list = list(self._obj.dims)
        third_dim_list.remove(mask_dims[0])
        third_dim_list.remove(mask_dims[1])
        layout = decimate(points) * \
                 dataset3d.to(hv.Image,
                              mask_dims,
                              'Value',
                              third_dim_list) * \
                 bounds + \
                 decimate(
                     hv.DynamicMap(
                         lambda bounds: hv.Points(
                             self._record_selections(bounds, points),
                             kdims=mask_dims
                         ),
                         streams=[box]
                     )
                 )
        return layout


    def __do_mask_changes(self, **kwargs):
        '''
        Does the desired changes to mask
        :param kwargs: array, if you want to pre select pixels,
                       initialize_mask, True if you want to initialize mask.
                                        True also Overrides array.
                                        False does nothing and overrides
                                        nothing
                       initialize_value, 1 or 0, the value you want to use in
                                        initialisation. Used if initialize_mask
                                        is True. Fallback value is 0.

        '''
        if not hasattr(self._obj, 'M'):
            MaskAccessor(self._obj)
        if 'initialize_mask' in kwargs:
            if kwargs['initialize_mask']:
                try:
                    if kwargs['initialize_value'] == 1:
                        self._obj.M.selected_ones()
                    else:
                        self._obj.M.selected_zeros()
                except KeyError:
                    self._obj.M.selected_zeros()
        elif 'array' in kwargs:
            self._obj.M.select(kwargs['array'])

    def _record_taps(self, first_coord, second_coord):
        '''
        Keeps taps and selected updated. Returns a holoviews element that it
        wants drawn.
        :param first_coord:
        :param second_coord:
        :return:
        '''
        assert isinstance(self._obj, xr.DataArray)
        wid = len(self._obj.coords[self._obj.M.dims[0]])
        hei = len(self._obj.coords[self._obj.M.dims[1]])
        if None not in [first_coord, second_coord]:
            first_rounded = round(first_coord)
            second_rounded = round(second_coord)
            if 0 <= second_rounded < wid and 0 <= first_rounded < hei:
                self._obj.M.select([second_rounded, first_rounded])
        try:
            ret = pd.DataFrame(columns=self._obj.M.dims,
                            data=self._obj.M.mask_to_pixels())
        except ValueError:
            ret = pd.DataFrame(columns=self._obj.M.dims,
                            data=[])
        return ret
        # return self._obj.M.mask_to_pixels()

    def _appropriate_amount_of_graphs(self):
        '''
        The amount of graphs in a picture is limited to ~4000.
        This makes sure we dont take more. We take every nth curve and
        skip the others. They stay in the selected -table though.
        TODO: test
        '''
        assert isinstance(self._obj, xr.DataArray)
        spectra_list = self._obj.M.to_list()
        how_many = len(spectra_list)
        max_amount = 3000
        if how_many <= max_amount:
            return spectra_list
        step = int(how_many / max_amount) + 1
        return spectra_list[0::step]

    def _new_choosing_spectre(self, bounds):
        '''
        Choosing spectre from image
        :param bounds: (left, bottom, right, top)
        :return: list of spectres, length under 3000
        '''
        pixel_list = self._obj.M.mask_to_pixels()
        spectra_list = self._obj.M.to_list()
        third_dim = self._obj.M.no_mask_dims[0]
        third_dim_data = self._obj.coords[third_dim].data
        
        for (i, spectre) in enumerate(spectra_list):
            drop = not self._is_in(spectre, third_dim_data, bounds)

            if drop:
                self._obj.M.unselect((pixel_list[i][0], pixel_list[i][1]))

        return self._appropriate_amount_of_graphs()

    @staticmethod
    def _is_in(spectre, third_dim_data, bounds):
        """
        Method to define if a graph goes through a box
        :param spectre: the graph y-data
        :param third_dim_data: the graph x-data
        :param bounds: the box in form of [x_low_left, y_ll, x_top_right, y_tr]
        :return: True or False
        """
        def _points_in(spectre, third_dim_data, bounds):
            x_low_left = bounds[0]
            y_low_left = bounds[1]
            x_top_right = bounds[2]
            y_top_right = bounds[3]
            for i, val in enumerate(spectre):
                y_value = val
                x_value = third_dim_data[i]
                if y_low_left <= y_value <= y_top_right \
                        and x_low_left <= x_value <= x_top_right:
                    return True
            return False

        def _goes_through(spectre, third_dim_data, bounds):
            x_low_left = bounds[0]
            y_low_left = bounds[1]
            x_top_right = bounds[2]
            y_top_right = bounds[3]
            line_1 = line((x_low_left, y_low_left), (x_low_left, y_top_right))
            # Left side of box
            line_2 = line((x_low_left, y_low_left), (x_top_right, y_low_left))
            # Bottom of box
            line_3 = line((x_low_left, y_top_right), (x_top_right, y_top_right))
            # Top of box
            line_4 = line((x_top_right, y_low_left), (x_top_right, y_top_right))
            # Right side of box
            for i in range(len(spectre) - 1):
                y_left = spectre[i]
                y_right = spectre[i + 1]
                x_left =  third_dim_data[i]
                x_right = third_dim_data[i + 1]
                line_5 = line((x_left, y_left), (x_right, y_right))
                intercept_1 = intersection(line_1, line_5)
                intercept_2 = intersection(line_2, line_5)
                intercept_3 = intersection(line_3, line_5)
                intercept_4 = intersection(line_4, line_5)
                if intercept_1:
                    y = intercept_1[1]
                    if y_low_left <= y <= y_top_right \
                            and min(y_left, y_right) <= y <= max(y_left,
                                                                 y_right):
                        return True

                if intercept_2:
                    x = intercept_2[0]
                    if x_low_left <= x <= x_top_right \
                            and min(x_left, x_right) <= x <= max(x_left,
                                                                 x_right):
                        return True

                if intercept_3:
                    x = intercept_3[0]
                    if x_low_left <= x <= x_top_right \
                            and min(x_left, x_right) <= x <= max(x_left,
                                                                 x_right):
                        return True

                if intercept_4:
                    y = intercept_4[1]
                    if y_low_left <= y <= y_top_right \
                            and min(y_left, y_right) <= y <= max(y_left,
                                                                 y_right):
                        return True

            return False

        points_in = _points_in(spectre,
                               third_dim_data,
                               bounds)
        goes_through = _goes_through(spectre,
                                     third_dim_data,
                                     bounds)

        return goes_through or points_in

    def _record_selections(self, bounds, points):
        '''
        Function that keeps track of what is selected. Keeps 'selected' -table
        up to date and returns the pixels that need to be drawn.
        TODO: redo
        '''
        assert isinstance(self._obj, xr.DataArray)
        attrs = {self._obj.M.dims[1]:(bounds[0],bounds[2]),
                 self._obj.M.dims[0]:(bounds[1],bounds[3])}
        ret = points.select(**attrs)
        arr = ret.data
        arr2 = []
        for value in arr:
            arr2.append([value[1], value[0]])
        self._obj.M.select(np.array(arr2))
        dims = self._obj.M.dims.copy()
        try:
            ret = pd.DataFrame(columns=dims,
                               data=self._obj.M.mask_to_pixels())
        except ValueError:
            ret = pd.DataFrame(columns=dims,
                               data=[])
        return ret

    def _make_dataset_opts(self):
        '''
        Makes Dataset options.
        '''
        data = []
        all_dims = self._obj.dims
        mask_dims = self._obj.M.dims.copy()
        for i, key in enumerate(all_dims):
            if key in mask_dims:
                data.append(range(self._obj.shape[i]))
            else:
                data.append(list(self._obj.coords[key].data))
        data.reverse()
        data.append(self._obj)
        # print('Data:' + str(data))
        data = tuple(data)
        # print(data)
        kdims = list(all_dims)
        kdims.reverse()
        ret = {'data':data,
               'kdims':kdims,
               'vdims':['Value']}
        # print(ret)
        return ret

    def histogram(self, band_dim, **kwargs):
        '''
        Histograms for each band, from above
        :param band_dim: Name of the band dimension
        :param kwargs:  flag: what to do for histogram values after calculation,
                            'linear' = nothing used if nothing else is given,
                            'cum' = calculate cumulative sums,
                            'log' = e-logarithm,
                            'log10' = 10-logarithm,
                            'log2' = 2-logarithm.
                        bin_edges: edges of the histogram bins. Overrides
                            n_bins,
                        visualize: Do you want to see the histogram?,
                        n_bins: Number of bins.
        :return: The image of histogram and histogram as numpy array and
                    bin edges
        '''
        # flag='log',
        # edges=None,
        # visualize=True,
        # n_bins=10
        if not len(self._obj.shape) == 3:
            raise ValueError('You can only calculate histogram for 3 ' +
                             'dimensional objects.')
        dims = list(self._obj.dims)
        dims.remove(band_dim)
        dims = tuple(dims)
        tmp = self._obj.stack(aa=dims)

        # By default, calculate simple counts for each band
        if 'flag' in kwargs:
            flag = self.__test_flags(kwargs['flag'])
        else:
            flag = 'linear'

        # If n_bins is given, we take it, at least for a while, fallback is 10
        if 'n_bins' in kwargs:
            n_bins = kwargs['n_bins']
        else:
            n_bins = math.sqrt(len(tmp.data[0]))

        # If bin_edges is given we drop the given n_bins and use the legth of
        # the bin_edges array instead.
        if 'bin_edges' in kwargs:
            bin_edges = kwargs['bin_edges']
            n_bins = len(bin_edges) - 1
        # if bin_edges is not given we use the n_bins to construct bin_edges
        elif np.max(self._obj.data) <= 1 and np.min(self._obj.data) >= 0:
            bin_edges = np.linspace(0, 1, n_bins + 1)
        else:
            bin_edges = np.linspace(np.min(self._obj.data),
                                    np.max(self._obj.data),
                                    n_bins + 1)
        # We take visualize in.
        if 'visualize' in kwargs:
            visualize = kwargs['visualize']
        else:
            visualize = True

        # Next we calculate the centers of bins
        centers = np.zeros(int(n_bins))
        for (i, _) in enumerate(centers):
            centers[i] = (bin_edges[i + 1] + bin_edges[i]) / 2

        # Next we calculate the actual histogram.
        counts = []
        for tmp_row in tmp:
            counts.append(np.histogram(tmp_row, bins=bin_edges, density=True)[0])
        counts_np = np.array(counts)
        if flag == 'cum':
            counts_np = np.cumsum(counts_np, axis=1)
        elif flag == 'log':
            counts_np = np.log(counts_np)
        elif flag == 'log2':
            counts_np = np.log2(counts_np)
        elif flag == 'log10':
            counts_np = np.log10(counts_np)
        elif flag == 'linear':
            pass
        layout = None
        if visualize:
            layout = self.__histogram_plot(centers, counts_np, band_dim)
        return layout, counts_np, bin_edges


    def __test_flags(self, flag):
        '''
        Helper function for histogram. Finds out if the flag is in the list.

        :param flag: tested flag
        :return: flag if its in the list, 'linear' if it is not
        '''
        flags = ['log', 'log2', 'log10', 'linear', 'cum']
        if flag not in flags:
            warnings.warn('Flag not recognized, using \'count\' instead.' +
                          ' Flag should be in ' + str(flags))
            return 'linear'
        return flag


    def __histogram_plot(self, centers, counts_np, band_dim):
        '''
        Plots the histogram
        :param centers: bin centers
        :param counts_np: histograms
        :param band_dim: dimension of bands
        :return: image of histogram
        '''
        # x_values = np.tile(self._obj.coords[band_dim].data, len(centers))
        # y_values = np.repeat(centers, len(self._obj.coords[band_dim].data))
        # values = counts_np.T.flatten()
        # table = hv.Table((x_values, y_values, values),
        #                 kdims=['Wavelength', 'Intensity'], vdims=['z'])
        # layout = [0,0]
        # layout[0] = hv.HeatMap(table).opts(plot=dict(tools=['hover'],
        #                                          colorbar=True,
        #                                           toolbar='above',
        #                                           show_title=False))
        # layout = table
        layout = hv.Image((self._obj.coords[band_dim].data,
                           centers,
                           counts_np.T),
                          kdims=['Wavelength',
                                 'Intensity'],
                          vdims=['z']).opts(plot=dict(tools=['hover'],
                                                      colorbar=True,
                                                      toolbar='above',
                                                      show_title=False,
                                                      backend='matplotlib'))
        return layout
