import slicer, vtk
import numpy as np

def convertToEQD2(dose, fraction):
    alpha_beta_ratio = 3
    eqd2 = dose * (dose/fraction + alpha_beta_ratio)/(2 + alpha_beta_ratio)
    return eqd2

def getDComponent(d1, d2, scale):
    return ((d1-d2)*scale)**2

def sigmoidDecay(x, r, a, b, c):
    return a/(1 + np.exp(b*(x-c*r)))

def doubleExpDecay(x, r, A1, a1, A2, a2):
    #Adjusting X
    return A1 * np.exp(-a1 * x) + A2 * np.exp(-a2 * x)
    
def getD(shape, center, r, sliceThickness=None, method="exp", dtype="float64", ):
    if not sliceThickness:
        sliceThickness = [1, 1, 1]
    x, y, z = np.ogrid[0:shape[0], 0:shape[1], 0:shape[2]]
    x_comp = getDComponent(x, center[0], sliceThickness[0])
    y_comp = getDComponent(y, center[1], sliceThickness[1])
    z_comp = getDComponent(z, center[2], sliceThickness[2])
        
    D = np.sqrt(x_comp + y_comp + z_comp).astype(dtype)

    #print("ogrid max x y z", np.max(x), np.max(y), np.max(z))

    if method == "exp":
        D = D - r
        D = np.maximum(D, np.zeros(D.shape))
    #print("Dmax, Dmin", np.max(D), np.min(D))
    return D

def getDWithNegatives(shape, center, r, sliceThickness=None, method="exp", dtype="float64", ):
    if not sliceThickness:
        sliceThickness = [1, 1, 1]
    x, y, z = np.ogrid[0:shape[0], 0:shape[1], 0:shape[2]]
    x_comp = getDComponent(x, center[0], sliceThickness[0])
    y_comp = getDComponent(y, center[1], sliceThickness[1])
    z_comp = getDComponent(z, center[2], sliceThickness[2])
        
    D = np.sqrt(x_comp + y_comp + z_comp).astype(dtype)

    #print("ogrid max x y z", np.max(x), np.max(y), np.max(z))

    if method == "exp":
        D = D - r
    #print("Dmax, Dmin", np.max(D), np.min(D))
    return D

def getMsSigmoid(D, r, prescription=1, a=100, b=0.46, c=1.15):
    M = sigmoidDecay(D, r, a, b, c) * prescription
    return M

def getMsDoubleExp(D, r, prescription=1, A1=65.7, a1=0.094, A2=50.8, a2=0.006):
    M = doubleExpDecay(D, r, A1, a1, A2, a2) /100 * prescription
    return M

def getMSphere(shape, center, radius, sliceThickness=None, prescription=1, method="exp", coeffs=None):
    modHU = np.ones(shape)
    D = getD(shape, center, radius, sliceThickness, method)
    if method == "sigmoid":
        #print("Sigmoid method used")
        Mm = np.multiply(modHU, getMsSigmoid(D, radius, prescription))
    elif method == "exp":
        #print("Double Exponential Used")
        if coeffs:
            Mm = np.multiply(modHU, getMsDoubleExp(D, radius, prescription, coeffs[0], coeffs[1], coeffs[2], coeffs[3]))
        else:
            Mm = np.multiply(modHU, getMsDoubleExp(D, radius, prescription))
    #print("Mm shape", Mm.shape)
    return Mm

def getMaskSurroundingPTV(volumeNode, ptv_ratio, ptvID="PTV"):
    H = slicer.util.arrayFromVolume(volumeNode) + 1024

    shape = H.shape
    H = np.ones(shape)

    #------[::-1] reverses the list
    sliceThickness = list(volumeNode.GetSpacing())[::-1]

    #Setting center to the center of PTV ----------------------------
    segmentId = ptvID
    segmentationNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
    seg = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId)

    #Putting segment in full size array
    seg_upsize = getFullSizeSegmentation(segmentId, shape)
    
    seg_idx = np.where(seg_upsize == 1)
    center = np.mean(np.asarray(seg_idx), axis=1)[::-1].astype(int)

    tumour_radius = np.mean((
        (np.max(seg_idx[0]) - np.min(seg_idx[0])) * sliceThickness[0], 
        (np.max(seg_idx[1]) - np.min(seg_idx[1])) * sliceThickness[1],
        (np.max(seg_idx[2]) - np.min(seg_idx[2])) * sliceThickness[2])) / 2
    
    sliceThickness = sliceThickness[::-1]
    max_x = center[0] + ptv_ratio * tumour_radius/sliceThickness[0]
    min_x = center[0] - ptv_ratio * tumour_radius/sliceThickness[0]

    max_y = center[1] + ptv_ratio * tumour_radius/sliceThickness[1]
    min_y = center[1] - ptv_ratio * tumour_radius/sliceThickness[1]

    max_z = center[2] + ptv_ratio * tumour_radius/sliceThickness[2]
    min_z = center[2] - ptv_ratio * tumour_radius/sliceThickness[2]
    
    max_x = np.min((int(max_x), shape[2]-1))
    max_y = np.min((int(max_y), shape[1]-1))
    max_z = np.min((int(max_z), shape[0]-1))
    
    min_x = np.max((int(min_x), 0))
    min_y = np.max((int(min_y), 0))
    min_z = np.max((int(min_z), 0))
    
    return(min_x, max_x, min_y, max_y, min_z, max_z)
    
    
def getMaskSurroundingPTVAbsolute(volumeNode, diameter, ptvID="PTV"):
    H = slicer.util.arrayFromVolume(volumeNode) + 1024

    shape = H.shape
    H = np.ones(shape)

    #------[::-1] reverses the list
    sliceThickness = list(volumeNode.GetSpacing())[::-1]

    #Setting center to the center of PTV ----------------------------
    segmentId = ptvID
    segmentationNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
    seg = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId, volumeNode)

    #Putting segment in full size array
    seg_upsize = getFullSizeSegmentation(segmentId, shape)
    
    seg_idx = np.where(seg_upsize == 1)
    center = np.mean(np.asarray(seg_idx), axis=1)[::-1].astype(int)

    tumour_radius = np.mean((
        (np.max(seg_idx[0]) - np.min(seg_idx[0])) * sliceThickness[0], 
        (np.max(seg_idx[1]) - np.min(seg_idx[1])) * sliceThickness[1],
        (np.max(seg_idx[2]) - np.min(seg_idx[2])) * sliceThickness[2])) / 2
    
    sliceThickness = sliceThickness[::-1]
    max_x = center[0] + diameter
    min_x = center[0] - diameter

    max_y = center[1] + diameter
    min_y = center[1] - diameter

    max_z = center[2] + diameter
    min_z = center[2] - diameter
    
    max_x = np.min((int(max_x), shape[2]-1))
    max_y = np.min((int(max_y), shape[1]-1))
    max_z = np.min((int(max_z), shape[0]-1))
    
    min_x = np.max((int(min_x), 0))
    min_y = np.max((int(min_y), 0))
    min_z = np.max((int(min_z), 0))
    
    return(min_x, max_x, min_y, max_y, min_z, max_z)

def convertToRTNode(node, parentNode):
    sh  = slicer.util.getNodesByClass("vtkMRMLSubjectHierarchyNode")[0]
    itemID = sh.GetItemByDataNode(node)
    parentID = sh.GetItemParent(sh.GetItemByDataNode(parentNode))
    sh.SetItemParent(itemID, parentID )
    sh.SetItemAttribute(parentID, 'DicomRtImport.DoseUnitName', "GY")
    sh.SetItemAttribute(parentID, 'DicomRtImport.DoseUnitValue', "1.0")
    node.SetAttribute("DicomRtImport.DoseVolume", "1")
    sh.RequestOwnerPluginSearch(itemID)
    sh.ItemModified(itemID)
    
def resampleScalarVolumeBrains(inputVolumeNode, referenceVolumeNode):
    # Set parameters
    parameters = {}
    parameters["inputVolume"] = inputVolumeNode
    try:
        outputModelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', "ResampledOriginalDose")
    except:
        outputModelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
    parameters["outputVolume"] = outputModelNode
    parameters["referenceVolume"] = referenceVolumeNode
    # Execute
    resampler = slicer.modules.brainsresample
    cliNode = slicer.cli.runSync(resampler, None, parameters)
    # Process results
    if cliNode.GetStatus() & cliNode.ErrorsMask:
        # error
        errorText = cliNode.GetErrorText()
        slicer.mrmlScene.RemoveNode(cliNode)
        raise ValueError("CLI execution failed: " + errorText)
    # success
    slicer.mrmlScene.RemoveNode(cliNode)
    return outputModelNode

def createVolumeNode(doseVolume, referenceNode, volumeName):
    doseNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', volumeName)
    doseNode.CopyOrientation(referenceNode)
    doseNode.SetSpacing(referenceNode.GetSpacing())
    doseNode.CreateDefaultDisplayNodes()
    displayNode = doseNode.GetDisplayNode()
    #Mmdn.SetLowerThreshold(-1020)
    #Mmdn.ApplyThresholdOn()
    #displayNode.SetAndObserveColorNodeID('vtkMRMLColorTableNodeReverseRainbow')
    displayNode.SetAndObserveColorNodeID('vtkMRMLColorTableNodeRainbow')
    slicer.util.updateVolumeFromArray(doseNode, doseVolume)
    #slicer.util.setSliceViewerLayers(background=volumeNode, foreground=MmNode)
    return doseNode

def getFullSizeSegmentation(segmentId, shape):
    segmentationNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
    referenceVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
    segarr = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId, referenceVolumeNode)
    #segImage = slicer.vtkOrientedImageData()
    #segmentationNode.GetBinaryLabelmapRepresentation(segmentId, segImage)
    #segImageExtent = segImage.GetExtent()
    #seg_upsize = np.zeros(shape)
    #seg_upsize[segImageExtent[4]:segImageExtent[5]+1, segImageExtent[2]:segImageExtent[3]+1, segImageExtent[0]:segImageExtent[1]+1] = segarr
    
    return segarr
    
def polydataToPoints(polydata):
    length = polydata.GetNumberOfPoints()
    points = np.zeros((length, 3))
    for i in range(length):
        polydata.GetPoint(i, points[i])
    return points

def getSegmentCenterRAS(segmentationNode, segmentId):
    seg = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId)
    # numpy array has voxel coordinates in reverse order (KJI instead of IJK)
    # and the array is cropped to minimum size in the segmentation
    mean_KjiCropped = [coords.mean() for coords in np.nonzero(seg)]

    # Get segmentation voxel coordinates
    segImage = slicer.vtkOrientedImageData()
    segmentationNode.GetBinaryLabelmapRepresentation(segmentId, segImage)
    segImageExtent = segImage.GetExtent()
    # origin of the array in voxel coordinates is determined by the start extent
    mean_Ijk = [mean_KjiCropped[2], mean_KjiCropped[1], mean_KjiCropped[0]] + np.array([segImageExtent[0], segImageExtent[2], segImageExtent[4]])

    # Get segmentation physical coordinates
    ijkToWorld = vtk.vtkMatrix4x4()
    segImage.GetImageToWorldMatrix(ijkToWorld)
    mean_World = [0, 0, 0, 1]
    ijkToWorld.MultiplyPoint(np.append(mean_Ijk,1.0), mean_World)
    mean_World = mean_World[0:3]

    # If segmentation node is transformed, apply that transform to get RAS coordinates
    transformWorldToRas = vtk.vtkGeneralTransform()
    slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(segmentationNode.GetParentTransformNode(), None, transformWorldToRas)
    mean_Ras = transformWorldToRas.TransformPoint(mean_World)
    return mean_Ras

def getDistanceArrayFromPoint(arr, center, percentile=0.1):
    distances = []
    for i in range(len(arr)):
        d = np.sqrt((arr[i][0] - center[0])**2 + (arr[i][1] - center[1])**2 + (arr[i][2] - center[2])**2)
        distances.append(d)
    #print(np.min(distances), np.max(distances))
    return np.percentile(distances, percentile)

def makeDVHReadable(fontsize=30):
    chart = slicer.util.getNodesByClass("vtkMRMLPlotChartNode")[0]
    chart.SetAxisLabelFontSize(fontsize)

def creatingCustomDoseMap(volumeNode, prescription, coeffs=None, ptvID="PTV"):
    #Getting parameters and calculating dose

    H = slicer.util.arrayFromVolume(volumeNode) + 1024

    shape = H.shape
    H = np.ones(shape)

    #------[::-1] reverses the list
    sliceThickness = list(volumeNode.GetSpacing())[::-1]

    #Setting center to the center of PTV ----------------------------
    segmentId = ptvID
    segmentationNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
    referenceVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
    seg = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId, referenceVolumeNode)

    #Putting segment in full size array
    seg_upsize = getFullSizeSegmentation(segmentId, shape)
    
    seg_idx = np.where(seg_upsize == 1)
    center = np.mean(np.asarray(seg_idx), axis=1)[::-1].astype(int)

    #PTV Radius ----------------------------------
    tumour_radius = np.mean((
            (np.max(seg_idx[0]) - np.min(seg_idx[0])) * sliceThickness[0], 
            (np.max(seg_idx[1]) - np.min(seg_idx[1])) * sliceThickness[1],
            (np.max(seg_idx[2]) - np.min(seg_idx[2])) * sliceThickness[2])) / 2
    
    
    #print("Tumour_radius", tumour_radius, "center", center)
    
    if coeffs is not None:
        Mm = getMSphere(shape[::-1], center, tumour_radius, sliceThickness=sliceThickness[::-1], prescription=prescription, method="exp", coeffs=coeffs)
    else:
        Mm = getMSphere(shape[::-1], center, tumour_radius, sliceThickness=sliceThickness[::-1], prescription=prescription, method="exp")
    
    Mm = np.transpose(Mm, (2, 1, 0))
    
    return Mm

def copySegment(inputId, outputId, segmentationNode):
    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)

    segmentEditorWidget.setSegmentationNode(segmentationNode)
    volumeNode = slicer.util.getNodesByClass("vtkMRMLVolumeNode")[0]
    segmentEditorWidget.setMasterVolumeNode(volumeNode)
    # Set overwrite mode: 0/1/2 -> overwrite all/visible/none
    segmentEditorNode.SetOverwriteMode(2) # i.e. "allow overlap" in UI
    # Get the segment IDs
    segmentationNode.AddSegmentFromClosedSurfaceRepresentation(vtk.vtkPolyData(), outputId)
    segid_tgt = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(outputId)
    segid_src = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(inputId)
    
    segmentEditorNode.SetSelectedSegmentID(segid_tgt)
    segmentEditorWidget.setActiveEffectByName("Logical operators")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("Operation","COPY") # change the operation here
    effect.setParameter("ModifierSegmentID",segid_src)
    effect.self().onApply()
    
def unionSegment(inputId, outputId, segmentationNode):
    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)

    segmentEditorWidget.setSegmentationNode(segmentationNode)
    volumeNode = slicer.util.getNodesByClass("vtkMRMLVolumeNode")[0]
    segmentEditorWidget.setMasterVolumeNode(volumeNode)
    # Set overwrite mode: 0/1/2 -> overwrite all/visible/none
    segmentEditorNode.SetOverwriteMode(2) # i.e. "allow overlap" in UI
    # Get the segment IDs
    segid_tgt = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(outputId)
    segid_src = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(inputId)
    
    segmentEditorNode.SetSelectedSegmentID(segid_tgt)
    segmentEditorWidget.setActiveEffectByName("Logical operators")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("Operation","UNION") # change the operation here
    effect.setParameter("ModifierSegmentID",segid_src)
    effect.self().onApply()
    
def subtractSegment(inputId, outputId, segmentationNode):
    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)

    segmentEditorWidget.setSegmentationNode(segmentationNode)
    volumeNode = slicer.util.getNodesByClass("vtkMRMLVolumeNode")[0]
    segmentEditorWidget.setMasterVolumeNode(volumeNode)
    # Set overwrite mode: 0/1/2 -> overwrite all/visible/none
    segmentEditorNode.SetOverwriteMode(2) # i.e. "allow overlap" in UI
    # Get the segment IDs
    segid_tgt = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(outputId)
    segid_src = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(inputId)
    
    segmentEditorNode.SetSelectedSegmentID(segid_tgt)
    segmentEditorWidget.setActiveEffectByName("Logical operators")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("Operation","SUBTRACT") # change the operation here
    effect.setParameter("ModifierSegmentID",segid_src)
    effect.self().onApply()
     
# Functions for calculating metrics

#For RAND2, actual R50 is 

def getR50(doseVolume, PTV, prescription):
    return getR_(doseVolume, PTV, prescription, 50)

def getR100(doseVolume, PTV, prescription):
    return getR_(doseVolume, PTV, prescription, 100)

def getR_(doseVolume, segmentId, prescription, isodose):
        #doseVolumeMask = np.isclose(doseVolume, prescription * 0.5)
    doseVolumeMask = doseVolume >= (prescription * (isodose / 100))
    volume50 = doseVolumeMask.sum()
    
    segmentationNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
    segarr = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId)
    segVolume = segarr.sum()
    
    return (volume50/segVolume)

def getIsodoseVolume(doseVolume, PTV, prescription, sliceThickness=None):
    #doseVolumeMask = np.isclose(doseVolume, prescription * 0.5)
    doseVolumeMask = doseVolume > (prescription / 2)
    return doseVolumeMask.astype(int)  
    
def getD2cm(doseVolume, segmentId):
    
    try:
        segmentationNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
        segarr = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, "PTV_Grown")
    except:
        segmentationNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
        segarr = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId)
        segVolume = segarr.sum()

        segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)

        segmentEditorWidget.setSegmentationNode(segmentationNode)
        volumeNode = slicer.util.getNodesByClass("vtkMRMLVolumeNode")[0]
        segmentEditorWidget.setMasterVolumeNode(volumeNode)

        # Set overwrite mode: 0/1/2 -> overwrite all/visible/none
        segmentEditorNode.SetOverwriteMode(2) # i.e. "allow overlap" in UI
        # Get the segment IDs
        segmentationNode.AddSegmentFromClosedSurfaceRepresentation(vtk.vtkPolyData(), "PTV_Grown")
        segid_tgt = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName('PTV_Grown')
        segid_src = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentId)


        # Running the effect of Creating a copy of the PTV:
        # The logical operation is set via strings
        # 'COPY', 'UNION', 'INTERSECT', 'SUBTRACT', 'INVERT', 'CLEAR', 'FILL'
        # (see: https://apidocs.slicer.org/master/SegmentEditorLogicalEffect_8py_source.html)    
        segmentEditorNode.SetSelectedSegmentID(segid_tgt)
        segmentEditorWidget.setActiveEffectByName("Logical operators")
        effect = segmentEditorWidget.activeEffect()
        effect.setParameter("Operation","COPY") # change the operation here
        effect.setParameter("ModifierSegmentID",segid_src)
        effect.self().onApply()

        #Growing the PTV by 20mm in each direction
        segmentEditorNode.SetSelectedSegmentID(segid_tgt)
        segmentEditorWidget.setActiveEffectByName("Margin")
        effect = segmentEditorWidget.activeEffect()
        effect.setParameter("Operation","GROW") # change the operation here
        effect.setParameter("MarginSizeMm", 20)
        effect.self().onApply()

        segarr = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, "PTV_Grown")
        
    
    seg_upsize = getFullSizeSegmentation("PTV_Grown", doseVolume.shape)
    
    inverse_segarr = 1-seg_upsize
    return (inverse_segarr * doseVolume).max()

def getV_X(doseVolume, segmentId, dose):
    seg_upsize = getFullSizeSegmentation(segmentId, doseVolume.shape)
    greaterThan20 = doseVolume > dose
    return (greaterThan20 * seg_upsize).sum()/seg_upsize.sum() * 100
    
def resizeVolume(newSpacing, originalNode, newNode, method="linear"):
    parameters = {}
    parameters["outputPixelSpacing"] = newSpacing
    parameters["interpolationType"] = method

    ##d = slicer.cli.createNode(slicer.modules.resamplescalarvolume, parameters=None)
    ##d.GetParameterName(0,1)
    parameters["InputVolume"] = originalNode
    parameters["OutputVolume"] = newNode

    # Execute
    resampleVolume = slicer.modules.resamplescalarvolume
    cliNode = slicer.cli.runSync(resampleVolume, None, parameters)

    if cliNode.GetStatus() & cliNode.ErrorsMask:
        # error
        errorText = cliNode.GetErrorText()
        slicer.mrmlScene.RemoveNode(cliNode)
        raise ValueError("CLI execution failed: " + errorText)
        
    return newNode
    
def addSegmentationToNodeFromNumpyArr(segmentationNode, numpyArr, name, referenceVolumeNode, color=[1,1,1]):
    tempVolumeNode = createVolumeNode(numpyArr, referenceVolumeNode, "TempNode")
    tempImageData = slicer.vtkSlicerSegmentationsModuleLogic.CreateOrientedImageDataFromVolumeNode(tempVolumeNode)
    slicer.mrmlScene.RemoveNode(tempVolumeNode)
    segmentationNode.AddSegmentFromBinaryLabelmapRepresentation(tempImageData, name, color)

def getVolume(shape, segmentId, sliceThickness):
    seg_upsize = getFullSizeSegmentation(segmentId, shape)
    seg_idx = np.where(seg_upsize == 1)
    num_voxels = len(seg_idx[0])
    return num_voxels * sliceThickness[0] * sliceThickness[1] * sliceThickness[2]