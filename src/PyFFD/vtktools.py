# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 15:32:22 2023

"""

import numpy as np


def array2string(array):
    return ' '.join([str(num) for num in array])


class VTKData:
    def __init__(self, name, numb, vals):
        self.name = name
        self.numb = numb
        if numb > 1:
            self.vals = vals.ravel()
        else:
            self.vals = vals


class VTRWriter():
    def __init__(self, xi, yi, zi):
        self.clearData()
        self.xi = xi
        self.yi = yi
        self.zi = zi

    def addCellData(self, varName, numb, data):
        cdnew = VTKData(varName, numb, data)
        self.cd = np.append(self.cd, cdnew)

    def addPointData(self, varName, numb, data):
        pdnew = VTKData(varName, numb, data)
        self.pd = np.append(self.pd, pdnew)

    def clearData(self):
        self.cd = np.empty(0, dtype=object)
        self.pd = np.empty(0, dtype=object)

    def VTRWriter(self, fileName):
        import xml.dom.minidom
        # Document and root element
        doc = xml.dom.minidom.Document()
        root_element = doc.createElementNS("VTK", "VTKFile")
        root_element.setAttribute("type", "RectilinearGrid")
        root_element.setAttribute("version", "0.1")
        root_element.setAttribute("byte_order", "LittleEndian")
        doc.appendChild(root_element)

        # Unstructured grid element
        RectilinearGrid = doc.createElementNS("VTK", "RectilinearGrid")
        extent = np.array(
            [0, len(self.xi)-1, 0, len(self.yi)-1, 0, len(self.zi)-1])
        RectilinearGrid.setAttribute("WholeExtent", array2string(extent))
        root_element.appendChild(RectilinearGrid)

        # Piece 0 (only one)
        piece = doc.createElementNS("VTK", "Piece")
        piece.setAttribute("Extent", array2string(extent))
        RectilinearGrid.appendChild(piece)

        ### Points ####
        points = doc.createElementNS("VTK", "Coordinates")
        piece.appendChild(points)

        # Point X Coordinates Data
        point_X_coords = doc.createElementNS("VTK", "DataArray")
        point_X_coords.setAttribute("type", "Float32")
        point_X_coords.setAttribute("Name", "X_COORDINATES")
        point_X_coords.setAttribute("NumberOfComponents", "1")
        point_X_coords.setAttribute("format", "ascii")
        points.appendChild(point_X_coords)

        point_X_coords_data = doc.createTextNode(array2string(self.xi))
        point_X_coords.appendChild(point_X_coords_data)

        # Point Y Coordinates Data
        point_Y_coords = doc.createElementNS("VTK", "DataArray")
        point_Y_coords.setAttribute("type", "Float32")
        point_Y_coords.setAttribute("Name", "Y_COORDINATES")
        point_Y_coords.setAttribute("NumberOfComponents", "1")
        point_Y_coords.setAttribute("format", "ascii")
        points.appendChild(point_Y_coords)

        point_Y_coords_data = doc.createTextNode(array2string(self.yi))
        point_Y_coords.appendChild(point_Y_coords_data)

        # Point Z Coordinates Data
        point_Z_coords = doc.createElementNS("VTK", "DataArray")
        point_Z_coords.setAttribute("type", "Float32")
        point_Z_coords.setAttribute("Name", "Z_COORDINATES")
        point_Z_coords.setAttribute("NumberOfComponents", "1")
        point_Z_coords.setAttribute("format", "ascii")
        points.appendChild(point_Z_coords)

        point_Z_coords_data = doc.createTextNode(array2string(self.zi))
        point_Z_coords.appendChild(point_Z_coords_data)

        #### Cell data  ####
        cell_data = doc.createElementNS("VTK", "CellData")
        piece.appendChild(cell_data)
        for ic in range(len(self.cd)):
            # Cell Data
            cell_data_array = doc.createElementNS("VTK", "DataArray")
            cell_data_array.setAttribute("Name", self.cd[ic].name)
            cell_data_array.setAttribute(
                "NumberOfComponents", str(self.cd[ic].numb))
            cell_data_array.setAttribute("type", "Float32")
            cell_data_array.setAttribute("format", "ascii")
            cell_data.appendChild(cell_data_array)
            cell_data_array_Data = doc.createTextNode(
                array2string(self.cd[ic].vals))
            cell_data_array.appendChild(cell_data_array_Data)

        #### Point data  ####
        point_data = doc.createElementNS("VTK", "PointData")
        piece.appendChild(point_data)
        for ic in range(len(self.pd)):
            # Point Data
            point_data_array = doc.createElementNS("VTK", "DataArray")
            point_data_array.setAttribute("Name", self.pd[ic].name)
            point_data_array.setAttribute(
                "NumberOfComponents", str(self.pd[ic].numb))
            point_data_array.setAttribute("type", "Float32")
            point_data_array.setAttribute("format", "ascii")
            point_data.appendChild(point_data_array)
            point_data_array_Data = doc.createTextNode(
                array2string(self.pd[ic].vals))
            point_data_array.appendChild(point_data_array_Data)

        # Write to file and exit
        outFile = open(fileName+'.vtr', 'w')
        doc.writexml(outFile, newl='\n')
        print("VTK: " + fileName + ".vtr written")
        outFile.close()
