import importlib
import logging
import os
import unittest

import ctk
import qt
import vtk

import slicer
from my_functions_SlicerExtension import *
import pandas as pd
import gc

#from models import Generator
import torch
from torch import nn


__all__ = ['MultiLesionLungSABRDosePrediction', 'MultiLesionLungSABRDosePredictionWidget', 'MultiLesionLungSABRDosePredictionLogic', 'MultiLesionLungSABRDosePredictionTest']
parent_dir = 'C:/Users/wanged/OneDrive - LHSC & St. Joseph\'s/Documents/LungTumourRadiotherapy/'
PATIENT_LIST_PATH = 'PATH_TO_CSV' #This is the CSV file that you would have used to generate the MRBs to train the model
SLICER_MRBS_PATH = 'PATH_TO_MRBS' #This is the folder where the MRBs are stored
COLOR_TABLE_PATH = 'PATH_TO/Isodose_ColorTable_RelativeCustom.ctbl')
GENERATOR_WEIGHT_PATH = 'GENERATOR_WEIGHT_PATH' #This is the path to the generator weights
GREEN = qt.QColor(12, 54, 25)
RED = qt.QColor(110, 7, 15)

class MultiLesionLungSABRDosePrediction:
  def __init__(self, parent):
    super().__init__()
    self.parent = parent
    self.moduleName = self.__class__.__name__

    parent.title = "MultiLesionLungSABRDosePrediction"
    parent.categories = ["Custom"]
    parent.dependencies = []
    parent.contributors = ["Andras Lasso (PerkLab, Queen's University), Steve Pieper (Isomics)"]
    parent.helpText = """
This module was created from a template and the help section has not yet been updated.
"""

    parent.acknowledgementText = """
This work is supported by NA-MIC, NAC, BIRN, NCIGT, and the Slicer Community. See <a>https://www.slicer.org</a> for details.
This work is partially supported by PAR-07-249: R01CA131718 NA-MIC Virtual Colonoscopy (See <a href=https://www.slicer.org>https://www.na-mic.org/Wiki/index.php/NA-MIC_NCBC_Collaboration:NA-MIC_virtual_colonoscopy</a>).
"""

    # Set module icon from Resources/Icons/<ModuleName>.png
    moduleDir = os.path.dirname(self.parent.path)
    for iconExtension in ['.svg', '.png']:
      iconPath = os.path.join(moduleDir, 'Resources/Icons', self.moduleName+iconExtension)
      if os.path.isfile(iconPath):
        parent.icon = qt.QIcon(iconPath)
        break

    # Add this test to the SelfTest module's list for discovery when the module
    # is created.  Since this module may be discovered before SelfTests itself,
    # create the list if it doesn't already exist.
    try:
      slicer.selfTests
    except AttributeError:
      slicer.selfTests = {}
    slicer.selfTests[self.moduleName] = self.runTest

  def getDefaultModuleDocumentationLink(self, docPage=None):
    """Return string that can be inserted into the application help text that contains
    link to the module's documentation in current Slicer version's documentation.
    The text is "For more information see the online documentation."
    If docPage is not specified then the link points to URL returned by :func:`slicer.app.moduleDocumentationUrl`.
    """
    if docPage:
      url = slicer.app.documentationBaseUrl + docPage
    else:
      url = slicer.app.moduleDocumentationUrl(self.moduleName)
    linkText = f'<p>For more information see the <a href="{url}">online documentation</a>.</p>'
    return linkText

  def runTest(self, msec=100, **kwargs):
    """
    :param msec: delay to associate with :func:`ScriptedLoadableModuleTest.delayDisplay()`.
    """
    # Name of the test case class is expected to be <ModuleName>Test
    module = importlib.import_module(self.__module__)
    className = self.moduleName + 'Test'
    try:
      TestCaseClass = getattr(module, className)
    except AttributeError:
      # Treat missing test case class as a failure; provide useful error message
      raise AssertionError(f'Test case class not found: {self.__module__}.{className} ')

    testCase = TestCaseClass()
    testCase.messageDelay = msec
    testCase.runTest(**kwargs)

class MultiLesionLungSABRDosePredictionWidget:
  def __init__(self, parent = None):
    """If parent widget is not specified: a top-level widget is created automatically;
    the application has to delete this widget (by calling widget.parent.deleteLater() to avoid memory leaks.
    """
    super().__init__()
    # Get module name by stripping 'Widget' from the class name
    self.moduleName = self.__class__.__name__
    if self.moduleName.endswith('Widget'):
      self.moduleName = self.moduleName[:-6]
    #self.developerMode = slicer.util.settingsValue('Developer/DeveloperMode', False, converter=slicer.util.toBool)
    self.developerMode = True
    if not parent:
      self.parent = slicer.qMRMLWidget()
      self.parent.setLayout(qt.QVBoxLayout())
      self.parent.setMRMLScene(slicer.mrmlScene)
    else:
      self.parent = parent
    self.layout = self.parent.layout()
    if not parent:
      self.setup()
      self.parent.show()
    slicer.app.moduleManager().connect(
      'moduleAboutToBeUnloaded(QString)', self._onModuleAboutToBeUnloaded)

  def resourcePath(self, filename):
    scriptedModulesPath = os.path.dirname(slicer.util.modulePath(self.moduleName))
    return os.path.join(scriptedModulesPath, 'Resources', filename)

  def cleanup(self):
    """Override this function to implement module widget specific cleanup.
    It is invoked when the signal `qSlicerModuleManager::moduleAboutToBeUnloaded(QString)`
    corresponding to the current module is emitted and just before a module is
    effectively unloaded.
    """
    pass

  def _onModuleAboutToBeUnloaded(self, moduleName):
    """This slot calls `cleanup()` if the module about to be unloaded is the
    current one.
    """
    if moduleName == self.moduleName:
      self.cleanup()
      slicer.app.moduleManager().disconnect(
        'moduleAboutToBeUnloaded(QString)', self._onModuleAboutToBeUnloaded)

  def setupInterface(self):
    def createVLayout(elements):
      rowLayout = qt.QVBoxLayout()
      for element in elements:
        rowLayout.addWidget(element)
      return rowLayout

    self.InputsCollapsibleButton = ctk.ctkCollapsibleButton()
    self.InputsCollapsibleButton.text = "Inputs"
    self.layout.addWidget(self.InputsCollapsibleButton)

    inputFormLayout = qt.QFormLayout(self.InputsCollapsibleButton)

    self.patientInput = createTextInput(self.InputsCollapsibleButton, inputFormLayout, "Patient ID: ", "LP_Test002")

    #dose
    #self.doseInput = createTextInput(self.InputsCollapsibleButton, inputFormLayout,
    #                                             "Dose: ", "60,60")
    #self.doseInput = createTextInput(self.InputsCollapsibleButton, inputFormLayout,
    #                                             "Dose: ", "60,55,60,60")
    #self.doseInput = createTextInput(self.InputsCollapsibleButton, inputFormLayout,
    #                                             "Dose: ", "60,60,60,60,60,60")
    self.doseInput = createTextInput(self.InputsCollapsibleButton, inputFormLayout,
                                                 "Dose: ", "24,24,24,24,24,24")

    #fraction
    #self.fractionInput = createTextInput(self.InputsCollapsibleButton, inputFormLayout,
     #                                            "Fraction: ", "8,8")
    #self.fractionInput = createTextInput(self.InputsCollapsibleButton, inputFormLayout,
    #                                             "Fraction: ", "8,5,8,8")
    #self.fractionInput = createTextInput(self.InputsCollapsibleButton, inputFormLayout,
    #                                             "Fraction: ", "8,8,8,8,8,8")
    self.fractionInput = createTextInput(self.InputsCollapsibleButton, inputFormLayout,
                                                 "Fraction: ", "1,1,1,1,1,1")

    #ptv
    #self.ptvInput = createTextInput(self.InputsCollapsibleButton, inputFormLayout,
    #                                             "PTV: ", "PTV_a,PTV_b")
    #self.ptvInput = createTextInput(self.InputsCollapsibleButton, inputFormLayout,
    #                                             "PTV: ", "PTV-SUP_LUNL1,PTV-INF_LUNL1,PTV60_LUNL2_3,PTV")
    #self.ptvInput = createTextInput(self.InputsCollapsibleButton, inputFormLayout,
    #                                             "PTV: ", "PTV,PTV1,PTV2,PTV1L,PTV2L,PTV3R")
    self.ptvInput = createTextInput(self.InputsCollapsibleButton, inputFormLayout,
                                                 "PTV: ", "PTVLUNG_R1,PTVLUNG_R2,PTVLUNG_R3,PTVLUNG_R4,PTVLUNG_L1,PTVLUNG_L2")


    #igtv
    #self.igtvPinput = createTextInput(self.InputsCollapsibleButton, inputFormLayout,
    #                                             "IGTV: ", "IGTV_a,IGTV_b")
    #self.igtvPinput = createTextInput(self.InputsCollapsibleButton, inputFormLayout,
    #                                             "IGTV: ", "IGTV-SUP_LUNL1,IGTV-INF_LUNL1,ITV_LUNL2_3,ITV")
    #self.igtvPinput = createTextInput(self.InputsCollapsibleButton, inputFormLayout,
    #                                             "IGTV: ", "IGTV,IGTV1,IGTV2,ITV1_L,ITV2_L,ITV3_R")
    self.igtvPinput = createTextInput(self.InputsCollapsibleButton, inputFormLayout,
                                                 "IGTV: ", "IGTVR1,IGTVR2,IGTVR3,IGTVR4,IGTVL1,IGTVL2")


    #Setting Values directly from CSV File
    self.updatePatient()


    self.createDoseCollapsibleButton = ctk.ctkCollapsibleButton()
    self.createDoseCollapsibleButton.text = "Create Dose"
    self.layout.addWidget(self.createDoseCollapsibleButton)

    doseFormLayout = qt.QFormLayout(self.createDoseCollapsibleButton)

    self.updatePatientButton = qt.QPushButton("Update Patient")
    self.updatePatientButton.connect('clicked()', self.updatePatient)

    self.initialProcessingButton = qt.QPushButton("Initial Preprocessing")
    self.initialProcessingButton.connect('clicked()', self.initialPreprocessing)

    self.createEstDoseButton = qt.QPushButton("Create Exp Dose")
    self.createEstDoseButton.connect('clicked()', self.createEstDose)

    self.createGANDoseButton = qt.QPushButton("Create Gan Dose")
    self.createGANDoseButton.connect('clicked()', self.createGANDose)

    self.calculateMetricsButton = qt.QPushButton("Calculate Metrics")
    self.calculateMetricsButton.connect('clicked()', self.calculateMetrics)

    self.addLesionButton = qt.QPushButton("Add Lesion")
    self.addLesionButton.connect("clicked()", self.addLesion)

    self.doAllButton = qt.QPushButton("Do All")
    self.doAllButton.connect("clicked()", self.doAll)

    doseFormLayout.addRow(createVLayout([self.updatePatientButton,
                                         #self.addLesionButton,
                                         self.initialProcessingButton,
                                         #self.createEstDoseButton,
                                         #self.createGANDoseButton,
                                         #self.calculateMetricsButton,
                                         self.doAllButton]))


    #Metric
    # self.metricInput = createTextInput(self.InputsCollapsibleButton, inputFormLayout,
    #                                              "DoseThreshold: ", "20")

    metrics = ['Lung V15', 'Lung CV14', 'Eso MaxD', 'SC MaxD', 'Heart MaxD', 'Airway MaxD', 'GV MaxD', 'CW MaxD']

    self.resultsTable = qt.QTableWidget(len(metrics), 2)
    self.InputsCollapsibleButton.layout().addWidget(self.resultsTable)
    self.resultsTable.setHorizontalHeaderLabels(["Constraint", "Prediction"])
    self.resultsTable.setVerticalHeaderLabels(metrics)

    self.lungV15 = qt.QTableWidgetItem()
    self.lungCV14 = qt.QTableWidgetItem()
    self.esoDmax = qt.QTableWidgetItem()
    self.scDmax = qt.QTableWidgetItem()
    self.heartDmax = qt.QTableWidgetItem()
    self.airwayDmax = qt.QTableWidgetItem()
    self.gvDmax = qt.QTableWidgetItem()
    self.cwDmax = qt.QTableWidgetItem()

    self.lungV15Constraint = qt.QTableWidgetItem()
    self.lungCV14Constraint = qt.QTableWidgetItem()
    self.esoDmaxConstraint = qt.QTableWidgetItem()
    self.scDmaxConstraint = qt.QTableWidgetItem()
    self.heartDmaxConstraint = qt.QTableWidgetItem()
    self.airwayDmaxConstraint = qt.QTableWidgetItem()
    self.gvDmaxConstraint = qt.QTableWidgetItem()
    self.cwDmaxConstraint = qt.QTableWidgetItem()

    self.resultsTable.setItem(0, 1, self.lungV15)
    self.resultsTable.setItem(1, 1, self.lungCV14)
    self.resultsTable.setItem(2, 1, self.esoDmax)
    self.resultsTable.setItem(3, 1, self.scDmax)
    self.resultsTable.setItem(4, 1, self.heartDmax)
    self.resultsTable.setItem(5, 1, self.airwayDmax)
    self.resultsTable.setItem(6, 1, self.gvDmax)
    self.resultsTable.setItem(7, 1, self.cwDmax)

    self.resultsTable.setItem(0, 0, self.lungV15Constraint)
    self.resultsTable.setItem(1, 0, self.lungCV14Constraint)
    self.resultsTable.setItem(2, 0, self.esoDmaxConstraint)
    self.resultsTable.setItem(3, 0, self.scDmaxConstraint)
    self.resultsTable.setItem(4, 0, self.heartDmaxConstraint)
    self.resultsTable.setItem(5, 0, self.airwayDmaxConstraint)
    self.resultsTable.setItem(6, 0, self.gvDmaxConstraint)
    self.resultsTable.setItem(7, 0, self.cwDmaxConstraint)


    # These constraints are taken from the planning document in Aria.
    # For all oars, use max dose constraint for 8 fractions, converted to EQD2
    # For chest wall, use 110% of 55/5
    # EQD2 of 3 used
    self.lungV15Constraint.setText(format(37, ".0f"))
    self.lungCV14Constraint.setText(format(1500, ".0f"))
    self.esoDmaxConstraint.setText(format(64, ".0f"))
    self.scDmaxConstraint.setText(format(45, ".0f"))
    self.heartDmaxConstraint.setText(format(81, ".0f"))
    self.airwayDmaxConstraint.setText(format(64, ".0f"))
    self.gvDmaxConstraint.setText(format(145, ".0f"))
    self.cwDmaxConstraint.setText(format(183, ".0f"))

  def setupDeveloperSection(self):
    if not self.developerMode:
      return

    def createHLayout(elements):
      rowLayout = qt.QHBoxLayout()
      for element in elements:
        rowLayout.addWidget(element)
      return rowLayout

    #
    # Reload and Test area
    # Used during development, but hidden when delivering
    # developer mode is turned off.

    self.reloadCollapsibleButton = ctk.ctkCollapsibleButton()
    self.reloadCollapsibleButton.text = "Reload && Test"
    self.layout.addWidget(self.reloadCollapsibleButton)
    reloadFormLayout = qt.QFormLayout(self.reloadCollapsibleButton)

    # reload button
    self.reloadButton = qt.QPushButton("Reload")
    self.reloadButton.toolTip = "Reload this module."
    self.reloadButton.name = "ScriptedLoadableModuleTemplate Reload"
    self.reloadButton.connect('clicked()', self.onReload)

    # reload and test button
    self.reloadAndTestButton = qt.QPushButton("Load Patient")
    self.reloadAndTestButton.toolTip = "Reload this module and then run the self tests."
    self.reloadAndTestButton.connect('clicked()', self.onReloadAndTest)

    # edit python source code
    self.editSourceButton = qt.QPushButton("Edit")
    self.editSourceButton.toolTip = "Edit the module's source code."
    self.editSourceButton.connect('clicked()', self.onEditSource)

    self.editModuleUiButton = None
    moduleUiFileName = self.resourcePath('UI/%s.ui' % self.moduleName)
    import os.path
    if os.path.isfile(moduleUiFileName):
      # Module UI file exists
      self.editModuleUiButton = qt.QPushButton("Edit UI")
      self.editModuleUiButton.toolTip = "Edit the module's .ui file."
      self.editModuleUiButton.connect('clicked()', lambda filename=moduleUiFileName: slicer.util.startQtDesigner(moduleUiFileName))

    # restart Slicer button
    # (use this during development, but remove it when delivering
    #  your module to users)
    self.restartButton = qt.QPushButton("Restart Slicer")
    self.restartButton.toolTip = "Restart Slicer"
    self.restartButton.name = "ScriptedLoadableModuleTemplate Restart"
    self.restartButton.connect('clicked()', slicer.app.restart)

    if self.editModuleUiButton:
      # There are many buttons, distribute them in two rows
      reloadFormLayout.addRow(createHLayout([self.reloadButton, self.reloadAndTestButton, self.restartButton]))
      reloadFormLayout.addRow(createHLayout([self.editSourceButton, self.editModuleUiButton]))
    else:
      reloadFormLayout.addRow(createHLayout([self.reloadButton, self.reloadAndTestButton, self.editSourceButton, self.restartButton]))

  def setup(self):
    self.resizedCtNode = None
    self.resampledOriginalDoseNode = None
    #self.originalDoseNode = None
    self.oarNode = None
    self.oarIgtvNode = None
    self.estDoseNode = None
    self.ganDoseNode = None
    self.originalSizeGanDoseNode = None
    self.new_x_min = 0
    self.new_x_max = 0
    self.new_y_min = 0
    self.new_y_max = 0
    self.new_z_min = 0
    self.new_z_max = 0
    self.isInitialPreprocessingDone = False

    self.fakeLesionCounter = 0

    # Instantiate and connect default widgets ...
    self.setupInterface()
    self.setupDeveloperSection()

    self.customColorTable = slicer.util.loadColorTable(COLOR_TABLE_PATH)

  def updatePatient(self):
    df = pd.read_csv(PATIENT_LIST_PATH)
    print(df)
    #
    dose = df[df["Patient"] == self.patientInput.text]["Dose"].values[0]
    fraction = df[df["Patient"] == self.patientInput.text]["Fraction"].values[0]
    ptv = df[df["Patient"] == self.patientInput.text]["PTVs"].values[0]
    igtv = df[df["Patient"] == self.patientInput.text]["IGTVs"].values[0]



    self.doseInput.setText(dose)
    self.fractionInput.setText(fraction)
    self.ptvInput.setText(ptv)
    self.igtvPinput.setText(igtv)

    if len(slicer.util.getNodesByClass("vtkMRMLVolumeNode")) > 0:
        referenceVolumeNode = slicer.util.getNodesByClass("vtkMRMLVolumeNode")[0]
        slicer.util.setSliceViewerLayers(background=referenceVolumeNode)

  def addLesion(self):
    self.fakeLesionCounter += 1
    referenceVolumeNode = slicer.util.getNodesByClass("vtkMRMLVolumeNode")[0]
    pointListNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
    point_Ras = [0, 0, 0, 1]
    pointListNode.GetNthFiducialWorldCoordinates(0, point_Ras) #First fiducial

    volumeRasToIjk = vtk.vtkMatrix4x4()
    referenceVolumeNode.GetRASToIJKMatrix(volumeRasToIjk)
    point_Ijk = [0, 0, 0, 1]
    volumeRasToIjk.MultiplyPoint(point_Ras, point_Ijk)
    point_Ijk = [ int(round(c)) for c in point_Ijk[0:3] ]

    segmentationNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")

    referenceVolumeArr = slicer.util.arrayFromVolume(referenceVolumeNode)
    fakeSegmentArr = np.zeros(referenceVolumeArr.shape)

    sliceThickness = list(referenceVolumeNode.GetSpacing())

    radius = 10
    max_x, max_y, max_z = point_Ijk[0] + int(radius / sliceThickness[0]), point_Ijk[1] + int(radius / sliceThickness[1]), point_Ijk[2] + int(radius / sliceThickness[2])
    min_x, min_y, min_z = point_Ijk[0] - int(radius / sliceThickness[0]), point_Ijk[1] - int(radius / sliceThickness[1]), point_Ijk[2] - int(radius / sliceThickness[2])

    fakeSegmentArr[min_z:max_z, min_y:max_y, min_x:max_x] = 1

    #pointListNode.RemoveAllMarkups()
    slicer.mrmlScene.RemoveNode(pointListNode)

    lesion_name = "FakeLesion" + str(self.fakeLesionCounter)

    addSegmentationToNodeFromNumpyArr(segmentationNode, fakeSegmentArr, lesion_name, referenceVolumeNode)

    self.doseInput.setText(self.doseInput.text + ",55")
    self.fractionInput.setText(self.fractionInput.text + ",5")
    self.ptvInput.setText(self.ptvInput.text + "," + lesion_name)
    self.igtvPinput.setText(self.igtvPinput.text + "," + lesion_name)

  def initialPreprocessing(self):
    if self.isInitialPreprocessingDone:
      print("Already Done Initial Preprocessing")
      pass
    else:
      print("Initial preprocess")
      self.resizedCtNode = None
      self.resampledOriginalDoseNode = None
      self.originalDoseNode = None
      self.oarNode = None
      self.oarIgtvNode = None
      self.estDoseNode = None
      self.ganDoseNode = None
      self.diffDoseNode = None
      self.originalSizeGanDoseNode = None
      self.customColorTable = slicer.util.loadColorTable(COLOR_TABLE_PATH)

      ptvs = self.ptvInput.text.split(",")
      igtvs = self.igtvPinput.text.split(",")

      num_lesions = len(ptvs)

      volumeNode = slicer.util.getNodesByClass("vtkMRMLVolumeNode")[0]
      referenceShape = slicer.util.arrayFromVolume(volumeNode).shape

      #Getting actual dose map
      # for i in range(10):
      #     origDoseNode  = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")[i]
      #     if "RTDOSE" in origDoseNode.GetName():
      #         break

      # if not self.resampledOriginalDoseNode:
      #     self.resampledOriginalDoseNode = resampleScalarVolumeBrains(origDoseNode, volumeNode, "ResampledOriginalDose")
      #     #Converting to RT dose node
      #     convertToRTNode(self.resampledOriginalDoseNode, volumeNode)

      segmentationNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
      segmentation = segmentationNode.GetSegmentation()
      IDs = vtk.vtkStringArray()
      segmentation.GetSegmentIDs(IDs)

      ##OARS with PTV

      self.oar_dict = {"Lung": 1,
                  "Heart": 7,
                  "Eso": 3,
                  "CW": 4,
                  "GV": 2,
                  "Trachea": 5,
                  "BT": 5,
                  "External": -1,
                  "SpinalCanal": 6,
                  }

      #print(referenceShape)
      lung_R_array = getFullSizeSegmentation(segmentation.GetSegmentIdBySegmentName("Lung_R"), referenceShape) * self.oar_dict["Lung"]
      lung_L_array = getFullSizeSegmentation(segmentation.GetSegmentIdBySegmentName("Lung_L"), referenceShape) * self.oar_dict["Lung"]
      heart_array = getFullSizeSegmentation(segmentation.GetSegmentIdBySegmentName("Heart"), referenceShape) * self.oar_dict["Heart"]
      esophagus_array = getFullSizeSegmentation(segmentation.GetSegmentIdBySegmentName("Esophagus"), referenceShape) * self.oar_dict["Eso"]
      chestWall_array = getFullSizeSegmentation(segmentation.GetSegmentIdBySegmentName("ChestWall"), referenceShape) * self.oar_dict["CW"]
      try:
        greatVessels_array = getFullSizeSegmentation(segmentation.GetSegmentIdBySegmentName("GreatVes"), referenceShape) * self.oar_dict["GV"]
      except:
        greatVessels_array = getFullSizeSegmentation(segmentation.GetSegmentIdBySegmentName("GV_Combined"),
                                                     referenceShape) * self.oar_dict["GV"]
      tracheaArray = getFullSizeSegmentation(segmentation.GetSegmentIdBySegmentName("Trachea"), referenceShape) * self.oar_dict["Trachea"]
      bronchialTreeArray = getFullSizeSegmentation(segmentation.GetSegmentIdBySegmentName("BronchialTree"), referenceShape) * self.oar_dict["BT"]
      externalArray = getFullSizeSegmentation(segmentation.GetSegmentIdBySegmentName("External"), referenceShape) * self.oar_dict["External"]
      spinalCanalArray = getFullSizeSegmentation(segmentation.GetSegmentIdBySegmentName("SpinalCanal"), referenceShape) * self.oar_dict["SpinalCanal"]

      ptv_arrays = []
      for i in range(num_lesions):
          print(ptvs[i])
          ptv_arrays.append(getFullSizeSegmentation(ptvs[i], referenceShape) * (10 + i))
      ptv_array = np.sum(ptv_arrays,axis = 0)

      # all_oars = [lung_R_array, lung_L_array, heart_array, esophagus_array, chestWall_array, greatVessels_array,
      #            tracheaArray, bronchialTreeArray, externalArray, spinalCanalArray]

      oars = externalArray.copy()
      oars[np.where(spinalCanalArray>0)] = self.oar_dict["SpinalCanal"]
      oars[np.where(heart_array>0)] = self.oar_dict["Heart"]
      oars[np.where(greatVessels_array>0)] = self.oar_dict["GV"]
      oars[np.where(esophagus_array>0)] = self.oar_dict["Eso"]
      oars[np.where(bronchialTreeArray>0)] = self.oar_dict["BT"]
      oars[np.where(tracheaArray>0)] = self.oar_dict["Trachea"]
      oars[np.where(chestWall_array>0)] = self.oar_dict["CW"]
      oars[np.where(lung_R_array>0)] = self.oar_dict["Lung"]
      oars[np.where(lung_L_array>0)] = self.oar_dict["Lung"]

      oars[np.where(ptv_array>0)] = 10

      oarNode = createVolumeNode(oars, volumeNode, "OARs")

      combined_lung_array = np.sum([lung_R_array, lung_L_array],axis=0)

      ptvNode = createVolumeNode(ptv_array, volumeNode, "PTVs")
      lungNode = createVolumeNode(combined_lung_array, volumeNode, "Lungs")

      # ogDoseArrResizedNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "ResizedOgDose")
      oarNodeResizedNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "ResizedOAR")
      lungResizedNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "ResizedLung")
      ptvResizedNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "ResizedPTVs")

      # ogDoseArrResizedNode = resizeVolume((3,3,3), self.resampledOriginalDoseNode, ogDoseArrResizedNode)
      oarNodeResizedNode = resizeVolume((3,3,3), oarNode, oarNodeResizedNode, "nearestNeighbor")
      lungResizedNode = resizeVolume((3,3,3), lungNode, lungResizedNode, "nearestNeighbor")
      ptvResizedNode = resizeVolume((3,3,3), ptvNode, ptvResizedNode, "nearestNeighbor")

      ##OARs with IGTV
      igtv_arrays = []
      for i in range(num_lesions):
          igtv_arrays.append(getFullSizeSegmentation(igtvs[i], referenceShape))
      igtv_array = np.sum(igtv_arrays,axis=0)

      oarsIgtv = externalArray.copy()
      oarsIgtv[np.where(spinalCanalArray>0)] = self.oar_dict["SpinalCanal"]
      oarsIgtv[np.where(heart_array>0)] = self.oar_dict["Heart"]
      oarsIgtv[np.where(greatVessels_array>0)] = self.oar_dict["GV"]
      oarsIgtv[np.where(esophagus_array>0)] = self.oar_dict["Eso"]
      oarsIgtv[np.where(bronchialTreeArray>0)] = self.oar_dict["BT"]
      oarsIgtv[np.where(tracheaArray>0)] = self.oar_dict["Trachea"]
      oarsIgtv[np.where(chestWall_array>0)] = self.oar_dict["CW"]
      oarsIgtv[np.where(lung_R_array>0)] = self.oar_dict["Lung"]
      oarsIgtv[np.where(lung_L_array>0)] = self.oar_dict["Lung"]
      oarsIgtv[np.where(igtv_array>0)] = 10

      #Deleting variables to free up memory
      del spinalCanalArray
      del heart_array
      del greatVessels_array
      del esophagus_array
      del bronchialTreeArray
      del tracheaArray
      del chestWall_array
      del lung_R_array
      del lung_L_array
      gc.collect()

      oarIGTVNode = createVolumeNode(oarsIgtv, volumeNode, "OARsIGTV")
      oarNodeResizedIGTVNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "ResizedIGTV")
      oarNodeResizedIGTVNode = resizeVolume((3,3,3), oarIGTVNode, oarNodeResizedIGTVNode, "nearestNeighbor")

      resizedCtNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "CT")
      resizedCtNode = resizeVolume((3,3,3), volumeNode, resizedCtNode)

      #Resampling SegmentationNode
      #transformedSegmNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
      #transformedSegmNode = resizeVolume((3,3,3), segmentationNode, transformedSegmNode)

      #Set reference geometry (resampled volume)
      #transformedSegmNode.SetReferenceImageGeometryParameterFromVolumeNode(resizedCtNode)


      # ogDose333 = slicer.util.arrayFromVolume(ogDoseArrResizedNode)
      oar333 = slicer.util.arrayFromVolume(oarNodeResizedNode)
      lung333 = slicer.util.arrayFromVolume(lungResizedNode)
      oarigtv333 = slicer.util.arrayFromVolume(oarNodeResizedIGTVNode)
      Ct333 = slicer.util.arrayFromVolume(resizedCtNode)
      Ct333 = np.clip(Ct333, -1024, 3071) #Clipping CT values 
      ptv333 = slicer.util.arrayFromVolume(ptvResizedNode)

      #Deleting Nodes

      slicer.mrmlScene.RemoveNode(oarNode)
      slicer.mrmlScene.RemoveNode(lungNode)
      slicer.mrmlScene.RemoveNode(ptvNode)
      # slicer.mrmlScene.RemoveNode(ogDoseArrResizedNode)
      slicer.mrmlScene.RemoveNode(oarNodeResizedNode)
      slicer.mrmlScene.RemoveNode(lungResizedNode)
      slicer.mrmlScene.RemoveNode(oarIGTVNode)
      slicer.mrmlScene.RemoveNode(oarNodeResizedIGTVNode)
      slicer.mrmlScene.RemoveNode(resizedCtNode)
      slicer.mrmlScene.RemoveNode(ptvResizedNode)


      not_zero_inds = np.where(lung333!=0)
      print(np.min(np.array(not_zero_inds), axis=1))
      print(np.max(np.array(not_zero_inds), axis=1))
      min_x, min_y, min_z = np.min(not_zero_inds[0]), np.min(not_zero_inds[1]), np.min(not_zero_inds[2])
      max_x, max_y, max_z = np.max(not_zero_inds[0]), np.max(not_zero_inds[1]), np.max(not_zero_inds[2])
      center_x = int((min_x + max_x)/2)
      center_y = int((min_y + max_y)/2)
      center_z = int((min_z + max_z)/2)

      print("center: ", center_x, center_y, center_z)

      new_x_min = center_x - 42
      new_x_max = center_x + 42

      new_y_min = center_y - 72
      new_y_max = center_y + 72

      new_z_min = center_z - 72
      new_z_max = center_z + 72

      print(new_x_min, new_x_max, new_y_min, new_y_max, new_z_min, new_z_max)
      if new_x_min < 0:
          adj = -new_x_min
          new_x_min += adj
          new_x_max += adj
      if new_y_min < 0:
          adj = -new_y_min
          new_y_min += adj
          new_y_max += adj
      if new_z_min < 0:
          adj = -new_z_min
          new_z_min += adj
          new_z_max += adj
      if new_x_max >= lung333.shape[0]:
          adj = new_x_max - lung333.shape[0]
          new_x_min -= adj
          new_x_max -= adj
      if new_y_max >= lung333.shape[1]:
          adj = new_y_max - lung333.shape[2]
          new_y_min -= adj
          new_y_max -= adj
      if new_z_max >= lung333.shape[2]:
          adj = new_z_max - lung333.shape[2]
          new_z_min -= adj
          new_z_max -= adj

      #Specifically for this test patient

      print(new_x_min, new_x_max, new_y_min, new_y_max, new_z_min, new_z_max)

      # ogDoseFinal = ogDose333[new_x_min:new_x_max, new_y_min:new_y_max, new_z_min:new_z_max]

      oarFinal = oar333[new_x_min:new_x_max, new_y_min:new_y_max, new_z_min:new_z_max]

      oarIgtvFinal = oarigtv333[new_x_min:new_x_max, new_y_min:new_y_max, new_z_min:new_z_max]


      CtFinal = Ct333[new_x_min:new_x_max, new_y_min:new_y_max, new_z_min:new_z_max]

      ptvFinal = ptv333[new_x_min:new_x_max, new_y_min:new_y_max, new_z_min:new_z_max]

      self.new_x_min = new_x_min
      self.new_x_max = new_x_max
      self.new_y_min = new_y_min
      self.new_y_max = new_y_max
      self.new_z_min = new_z_min
      self.new_z_max = new_z_max

      resizedReferenceNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "ResizedReference")
      resizedReferenceNode = resizeVolume((3,3,3), volumeNode, resizedReferenceNode)

      # if not self.originalDoseNode:
      #     self.originalDoseNode = createVolumeNode(ogDoseFinal, resizedReferenceNode, "OriginalDose")
      if not self.oarNode:
          self.oarNode = createVolumeNode(oarFinal, resizedReferenceNode, "oarPTV")
      if not self.oarIgtvNode:
          self.oarIgtvNode = createVolumeNode(oarIgtvFinal, resizedReferenceNode, "oarIgtv")
      if not self.resizedCtNode:
          self.resizedCtNode = createVolumeNode(CtFinal, resizedReferenceNode, "CT")


      #Making new Segmentation after resizing
      transformedSegmNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
      transformedSegmNode.AddNodeReferenceRole("display")

      addSegmentationToNodeFromNumpyArr(transformedSegmNode, (oarIgtvFinal == 1).astype("int"), "Lungs", resizedCtNode, [0.3, 0.9, 1])
      #addSegmentationToNodeFromNumpyArr(transformedSegmNode, (oarFinal == 2).astype("int") , "Heart", resizedCtNode,  [0.3, 0.2, 0])
      #addSegmentationToNodeFromNumpyArr(transformedSegmNode, (oarFinal == 3).astype("int"), "Eso", resizedCtNode,  [0.9, 0.6, 1])
      for i in range(num_lesions):
          addSegmentationToNodeFromNumpyArr(transformedSegmNode, (ptvFinal == (10 + i)).astype("int"), "PTV" + str(i), resizedCtNode,  [0.5, 0.5, 1])

      displayNode1 = self.oarNode.GetDisplayNode()
      displayNode1.SetAndObserveColorNodeID('vtkMRMLColorTableNodeGrey')
      displayNode1 = self.oarIgtvNode.GetDisplayNode()
      displayNode1.SetAndObserveColorNodeID('vtkMRMLColorTableNodeGrey')
      displayNode1 = self.resizedCtNode.GetDisplayNode()
      displayNode1.SetAndObserveColorNodeID('vtkMRMLColorTableNodeGrey')

      segmentationNode.GetDisplayNode().SetVisibility(False)

      dn = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationDisplayNode")
      transformedSegmNode.AddAndObserveDisplayNodeID(dn.GetID())
      transformedSegmNode.GetDisplayNode().SetVisibility(False)

      slicer.mrmlScene.RemoveNode(resizedReferenceNode)

      
      self.g = AttentionGenerator(3, 1)
    
      if torch.cuda.is_available():
          device = torch.device("cuda")
          print("Using GPU")
      else:
          device = torch.device("cpu")
          print("Using CPU")
      self.g.to(device)
      if torch.cuda.is_available():
          self.g.load_state_dict(torch.load(GENERATOR_WEIGHT_PATH))
      else:
          self.g.load_state_dict(torch.load(GENERATOR_WEIGHT_PATH, map_location="cpu"))
      self.g.eval()
      self.device = device
      
      self.isInitialPreprocessingDone = True

  def createEstDose(self):
    print("Create estimated dose")
    prescription = self.doseInput.text.split(",")
    frac = self.fractionInput.text.split(",")
    ptvs = self.ptvInput.text.split(",")

    num_lesions = len(ptvs)

    prescription_list = prescription
    frac_list = frac

    #Convert prescription to EQD2
    for i in range(len(prescription_list)):
        prescription_list[i] = convertToEQD2(float(prescription_list[i]), int(frac_list[i]))

    A1 = 91.9
    a1 = 0.104
    A2 = 8.09
    a2 = 0.027

    volumeNode = slicer.util.getNodesByClass("vtkMRMLVolumeNode")[0]

    #Getting dose maps for all lesions
    Mm = []

    for i in range(num_lesions):
        Mm.append(creatingCustomDoseMap(volumeNode, float(prescription_list[i]), coeffs=[A1,a1,A2,a2], ptvID=ptvs[i]))

    referenceShape = Mm[0].shape

    Mm_summed = np.zeros(referenceShape)

    for i in range(num_lesions):
        Mm_summed = Mm_summed + Mm[i]

    estDoseNode = createVolumeNode(Mm_summed, volumeNode, "CalculatedDoseNode")

    #resize volume node to 3x3x3
    estDoseResizedNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "tempEstDose")
    estDoseResizedNode = resizeVolume((3,3,3), estDoseNode, estDoseResizedNode)

    #estDoseNode
    estDose333 = slicer.util.arrayFromVolume(estDoseResizedNode)

    estDoseFinal = estDose333[self.new_x_min:self.new_x_max, self.new_y_min:self.new_y_max, self.new_z_min:self.new_z_max]

    slicer.mrmlScene.RemoveNode(estDoseNode)
    slicer.mrmlScene.RemoveNode(estDoseResizedNode)

    resizedReferenceNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "ResizedReference")
    resizedReferenceNode = resizeVolume((3,3,3), volumeNode, resizedReferenceNode)

    if not self.estDoseNode:
        self.estDoseNode = createVolumeNode(estDoseFinal, resizedReferenceNode, "EstDose")
    else:
        slicer.mrmlScene.RemoveNode(self.estDoseNode)
        self.estDoseNode = createVolumeNode(estDoseFinal, resizedReferenceNode, "EstDose")

    slicer.mrmlScene.RemoveNode(resizedReferenceNode)

    slicer.util.setSliceViewerLayers(background=self.resizedCtNode, foreground=self.estDoseNode, foregroundOpacity=0.7)

  def createGANDose(self):
    print("Create GAN Dose")
    volumeNode = slicer.util.getNodesByClass("vtkMRMLVolumeNode")[0]


    est_dose = torch.tensor(slicer.util.arrayFromVolume(self.estDoseNode)).unsqueeze(0).unsqueeze(0).float().to(self.device)
    oars = torch.tensor(slicer.util.arrayFromVolume(self.oarNode)).unsqueeze(0).unsqueeze(0).float().to(self.device)
    ct = torch.tensor(slicer.util.arrayFromVolume(self.resizedCtNode)).unsqueeze(0).unsqueeze(0).float().to(self.device)

    #print(est_dose.shape)
    #print(oars.shape)

    print("input to gan shape", est_dose.shape, oars.shape)

    fake_dose = self.g(torch.cat((est_dose, ct), dim=1), oars).cpu().detach().numpy()[0,0,:,:,:]
    fake_dose = np.clip(fake_dose, 0, None)

    resizedReferenceNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "ResizedReference")
    resizedReferenceNode = resizeVolume((3,3,3), volumeNode, resizedReferenceNode)

    if self.ganDoseNode:
        slicer.mrmlScene.RemoveNode(self.ganDoseNode)

    self.ganDoseNode = createVolumeNode(fake_dose, resizedReferenceNode, "GeneratedDose")

    if self.originalSizeGanDoseNode:
        slicer.mrmlScene.RemoveNode(self.originalSizeGanDoseNode)

    referenceShape = slicer.util.arrayFromVolume(volumeNode).shape
    uncroppedDose = np.zeros(referenceShape)

    
    # print(uncroppedDose[new_x_min:new_x_max, new_y_min:new_y_max, new_z_min:new_z_max].shape, syntheticDose.shape)
    
    
    if self.new_x_max - self.new_x_min < 92:
        uncroppedDose[self.new_x_min:self.new_x_max, self.new_y_min:self.new_y_max, self.new_z_min:self.new_z_max] = fake_dose[:self.new_x_max, :, :]
    else:
        uncroppedDose[self.new_x_min:self.new_x_max, self.new_y_min:self.new_y_max, self.new_z_min:self.new_z_max] = fake_dose

    resizedReferenceNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "ResizedReference")
    resizedReferenceNode = resizeVolume((3,3,3), volumeNode, resizedReferenceNode)
    uncroppedSyntheticDoseNode = createVolumeNode(uncroppedDose, resizedReferenceNode, "GeneratedDose")

    self.originalSizeGanDoseNode  = resampleScalarVolumeBrains(uncroppedSyntheticDoseNode, volumeNode, "ResampledGANDose")


    convertToRTNode(self.originalSizeGanDoseNode, volumeNode)

    displayNode = self.originalSizeGanDoseNode.GetDisplayNode()
    displayNode.SetAndObserveColorNodeID(self.customColorTable.GetID())

    displayNode.AutoWindowLevelOff()
    displayNode.SetWindowLevelMinMax(0, np.max(fake_dose))


    colorLegendDisplayNode = slicer.modules.colors.logic().AddDefaultColorLegendDisplayNode(self.originalSizeGanDoseNode)
    colorLegendDisplayNode.SetNumberOfLabels(10)
    colorLegendDisplayNode.SetMaxNumberOfColors(255)
    colorLegendDisplayNode.SetTitleText("")
    colorLegendDisplayNode.SetLabelFormat('%.0f')
    colorLegendDisplayNode.SetUseColorNamesForLabels(False)

    slicer.util.setSliceViewerLayers(background=volumeNode, foreground=self.originalSizeGanDoseNode, foregroundOpacity=0.5)

  def calculateMetrics(self):
    print("Calculate Metrics")

    oarsIgtv = slicer.util.arrayFromVolume(self.oarIgtvNode)

    #ganDose = slicer.util.arrayFromVolume(self.originalDoseNode)
    ganDose = slicer.util.arrayFromVolume(self.ganDoseNode)
    sliceThickness = list(self.ganDoseNode.GetSpacing())

    lungV15 = getV_X(ganDose, oarsIgtv, self.oar_dict["Lung"], 15)
    lungCV14 = getCV(ganDose, oarsIgtv, self.oar_dict["Lung"], 14, sliceThickness)

    # numVoxels5cc =  int(5 * 1000 / (sliceThickness[0] * sliceThickness[1] * sliceThickness[2]))
    # numVoxels10cc = int(10 * 1000 / (sliceThickness[0] * sliceThickness[1] * sliceThickness[2]))
    # numVoxels15cc = int(15 * 1000 / (sliceThickness[0] * sliceThickness[1] * sliceThickness[2]))

    # esoD5cc = getDmaxForGivenNumVoxels(ganDose, oarsIgtv, self.oar_dict["Eso"], numVoxels5cc)
    # heartD15cc = getDmaxForGivenNumVoxels(ganDose, oarsIgtv, self.oar_dict["Heart"], numVoxels15cc)
    # airwayD5cc = getDmaxForGivenNumVoxels(ganDose, oarsIgtv, self.oar_dict["Trachea"], numVoxels5cc)
    # gvd10cc = getDmaxForGivenNumVoxels(ganDose, oarsIgtv, self.oar_dict["GV"], numVoxels10cc)
    # cwd5cc = getDmaxForGivenNumVoxels(ganDose, oarsIgtv, self.oar_dict["CW"], numVoxels5cc)

    esoDmax = getDmax(ganDose, oarsIgtv, self.oar_dict["Eso"],)
    scDmax = getDmax(ganDose, oarsIgtv, self.oar_dict["SpinalCanal"],)
    heartDmax = getDmax(ganDose, oarsIgtv, self.oar_dict["Heart"],)
    airwayDmax = getDmax(ganDose, oarsIgtv, self.oar_dict["Trachea"],)
    gvdDmax = getDmax(ganDose, oarsIgtv, self.oar_dict["GV"],)
    cwDmax = getDmax(ganDose, oarsIgtv, self.oar_dict["CW"])

    self.lungV15.setText(format(lungV15, ".0f"))
    self.lungCV14.setText(format(lungCV14, ".0f"))
    self.esoDmax.setText(format(esoDmax, ".0f"))
    self.scDmax.setText(format(scDmax, ".0f"))
    self.heartDmax.setText(format(heartDmax, ".0f"))
    self.airwayDmax.setText(format(airwayDmax, ".0f"))
    self.gvDmax.setText(format(gvdDmax, ".0f"))
    self.cwDmax.setText(format(cwDmax, ".0f"))

    boldFont = qt.QFont()
    boldFont.setBold(True)

    if float(self.lungV15.text()) <= float(self.lungV15Constraint.text()):
        self.lungV15.setBackground(GREEN)
    else:
        self.lungV15.setBackground(RED)
    self.lungV15.setFont(boldFont)

    if float(self.lungCV14.text()) >= float(self.lungCV14Constraint.text()): #this constraint is special because it should be greater than
        self.lungCV14.setBackground(GREEN)
    else:
        self.lungCV14.setBackground(RED)
    self.lungCV14.setFont(boldFont)

    if float(self.esoDmax.text()) <= float(self.esoDmaxConstraint.text()):
        self.esoDmax.setBackground(GREEN)
    else:
        self.esoDmax.setBackground(RED)
    self.esoDmax.setFont(boldFont)

    if float(self.scDmax.text()) <= float(self.scDmaxConstraint.text()):
        self.scDmax.setBackground(GREEN)
    else:
        self.scDmax.setBackground(RED)
    self.scDmax.setFont(boldFont)

    if float(self.heartDmax.text()) <= float(self.heartDmaxConstraint.text()):
        self.heartDmax.setBackground(GREEN)
    else:
        self.heartDmax.setBackground(RED)
    self.heartDmax.setFont(boldFont)

    if float(self.airwayDmax.text()) <= float(self.airwayDmaxConstraint.text()):
        self.airwayDmax.setBackground(GREEN)
    else:
        self.airwayDmax.setBackground(RED)
    self.airwayDmax.setFont(boldFont)

    if float(self.gvDmax.text()) <= float(self.gvDmaxConstraint.text()):
        self.gvDmax.setBackground(GREEN)
    else:
        self.gvDmax.setBackground(RED)
    self.gvDmax.setFont(boldFont)

    if float(self.cwDmax.text()) <= float(self.cwDmaxConstraint.text()):
        self.cwDmax.setBackground(GREEN)
    else:
        self.cwDmax.setBackground(RED)
    self.cwDmax.setFont(boldFont)


    # doseThreshold = float(self.metricInput.text)
    # #realDose = slicer.util.arrayFromVolume(self.originalDoseNode)
    # estDose = slicer.util.arrayFromVolume(self.estDoseNode)
    # oarsIgtv = slicer.util.arrayFromVolume(self.oarIgtvNode)
    # ganDose = slicer.util.arrayFromVolume(self.ganDoseNode)
    #
    # realDose = ganDose
    #
    # lung_real, lung_est, lung_gan, lung_est_diff, lung_gan_diff = getV_Xdiff(realDose, estDose, ganDose, oarsIgtv, 1, doseThreshold)
    # heart_real, heart_est, heart_gan, heart_est_diff, heart_gan_diff = getV_Xdiff(realDose, estDose, ganDose, oarsIgtv, 2, doseThreshold)
    # eso_real, eso_est, eso_gan, eso_est_diff, eso_gan_diff = getV_Xdiff(realDose, estDose, ganDose, oarsIgtv, 3, doseThreshold)
    # ptv_real, ptv_est, ptv_gan, ptv_est_diff, ptv_gan_diff = getV_Xdiff(realDose, estDose, ganDose, oarsIgtv, 4, doseThreshold)
    #
    # self.Pred_Lung.setText(format(lung_real, ".3f"))
    #
    # self.Pred_Heat.setText(format(heart_real, ".3f"))
    #
    # self.Pred_Eso.setText(format(eso_real, ".3f"))
    #
    # self.Pred_PTV.setText(format(ptv_real, ".3f"))

  def doAll(self):
    #self.initialPreprocessing()
    self.createEstDose()
    self.createGANDose()
    self.calculateMetrics()

  def onReload(self):
    """
    Reload scripted module widget representation.
    """

    # Print a clearly visible separator to make it easier
    # to distinguish new error messages (during/after reload)
    # from old ones.
    print('\n' * 2)
    print('-' * 30)
    print('Reloading module: '+self.moduleName)
    print('-' * 30)
    print('\n' * 2)

    print(self.moduleName)
    slicer.util.reloadScriptedModule(self.moduleName)
    #slicer.util.reloadScriptedModule("DoseEstimationModule")

  def onReloadAndTest(self, **kwargs):
    """Reload scripted module widget representation and call :func:`ScriptedLoadableModuleTest.runTest()`
    passing ``kwargs``.
    """
    slicer.mrmlScene.Clear(0)
    slicer.util.loadScene(SLICER_MRBS_PATH + self.patientInput.text + ".mrb")

    originalCT = slicer.util.getNodesByClass("vtkMRMLVolumeNode")[0]

    #slicer.util.setSliceViewerLayers(background=self.resizedCtNode, foreground=self.originalDoseNode, foregroundOpacity=0.7)
    slicer.util.setSliceViewerLayers(originalCT)

  def onEditSource(self):
    filePath = slicer.util.modulePath(self.moduleName)
    qt.QDesktopServices.openUrl(qt.QUrl("file:///"+filePath, qt.QUrl.TolerantMode))

class MultiLesionLungSABRDosePredictionLogic:
  def __init__(self, parent = None):
    super().__init__()
    # Get module name by stripping 'Logic' from the class name
    self.moduleName = self.__class__.__name__
    if self.moduleName.endswith('Logic'):
      self.moduleName = self.moduleName[:-5]

    # If parameter node is singleton then only one parameter node
    # is allowed in a scene.
    # Derived classes can set self.isSingletonParameterNode = False
    # to allow having multiple parameter nodes in the scene.
    self.isSingletonParameterNode = True

  def getParameterNode(self):
    """
    Return the first available parameter node for this module
    If no parameter nodes are available for this module then a new one is created.
    """
    if self.isSingletonParameterNode:
      parameterNode = slicer.mrmlScene.GetSingletonNode(self.moduleName, "vtkMRMLScriptedModuleNode")
      if parameterNode:
        # After close scene, ModuleName attribute may be removed, restore it now
        if parameterNode.GetAttribute("ModuleName") != self.moduleName:
          parameterNode.SetAttribute("ModuleName", self.moduleName)
        return parameterNode
    else:
      numberOfScriptedModuleNodes =  slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLScriptedModuleNode")
      for nodeIndex in range(numberOfScriptedModuleNodes):
        parameterNode  = slicer.mrmlScene.GetNthNodeByClass( nodeIndex, "vtkMRMLScriptedModuleNode" )
        if parameterNode.GetAttribute("ModuleName") == self.moduleName:
          return parameterNode
    # no parameter node was found for this module, therefore we add a new one now
    parameterNode = slicer.mrmlScene.AddNode(self.createParameterNode())
    return parameterNode

  def getAllParameterNodes(self):
    """
    Return a list of all parameter nodes for this module
    Multiple parameter nodes are useful for storing multiple parameter sets in a single scene.
    """
    foundParameterNodes = []
    numberOfScriptedModuleNodes =  slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLScriptedModuleNode")
    for nodeIndex in range(numberOfScriptedModuleNodes):
      parameterNode  = slicer.mrmlScene.GetNthNodeByClass( nodeIndex, "vtkMRMLScriptedModuleNode" )
      if parameterNode.GetAttribute("ModuleName") == self.moduleName:
        foundParameterNodes.append(parameterNode)
    return foundParameterNodes

  def createParameterNode(self):
    """
    Create a new parameter node
    The node is of vtkMRMLScriptedModuleNode class. Module name is added as an attribute to allow filtering
    in node selector widgets (attribute name: ModuleName, attribute value: the module's name).
    This method can be overridden in derived classes to create a default parameter node with all
    parameter values set to their default.
    """
    if slicer.mrmlScene is None:
      return

    node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScriptedModuleNode")
    node.UnRegister(None) # object is owned by the Python variable now
    if self.isSingletonParameterNode:
      node.SetSingletonTag( self.moduleName )
    # Add module name in an attribute to allow filtering in node selector widgets
    # Note that SetModuleName is not used anymore as it would be redundant with the ModuleName attribute.
    node.SetAttribute( "ModuleName", self.moduleName )
    node.SetName(slicer.mrmlScene.GenerateUniqueName(self.moduleName))
    return node

class MultiLesionLungSABRDosePredictionTest(unittest.TestCase):
  """
  Base class for module tester class.
  Setting messageDelay to something small, like 50ms allows
  faster development time.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # takeScreenshot default parameters
    self.enableScreenshots = False
    self.screenshotScaleFactor = 1.0

  def delayDisplay(self,message,requestedDelay=None,msec=None):
    """
    Display messages to the user/tester during testing.
    By default, the delay is 50ms.
    The function accepts the keyword arguments ``requestedDelay`` or ``msec``. If both
    are specified, the value associated with ``msec`` is used.
    This method can be temporarily overridden to allow tests running
    with longer or shorter message display time.
    Displaying a dialog and waiting does two things:
    1) it lets the event loop catch up to the state of the test so
    that rendering and widget updates have all taken place before
    the test continues and
    2) it shows the user/developer/tester the state of the test
    so that we'll know when it breaks.
    Note:
    Information that might be useful (but not important enough to show
    to the user) can be logged using logging.info() function
    (printed to console and application log) or logging.debug()
    function (printed to application log only).
    Error messages should be logged by logging.error() function
    and displayed to user by slicer.util.errorDisplay function.
    """
    if hasattr(self, "messageDelay"):
      msec = self.messageDelay
    if msec is None:
      msec = requestedDelay
    if msec is None:
      msec = 100

    slicer.util.delayDisplay(message, msec)

  def takeScreenshot(self,name,description,type=-1):
    """ Take a screenshot of the selected viewport and store as and
    annotation snapshot node. Convenience method for automated testing.
    If self.enableScreenshots is False then only a message is displayed but screenshot
    is not stored. Screenshots are scaled by self.screenshotScaleFactor.
    :param name: snapshot node name
    :param description: description of the node
    :param type: which viewport to capture. If not specified then captures the entire window.
      Valid values: slicer.qMRMLScreenShotDialog.FullLayout,
      slicer.qMRMLScreenShotDialog.ThreeD, slicer.qMRMLScreenShotDialog.Red,
      slicer.qMRMLScreenShotDialog.Yellow, slicer.qMRMLScreenShotDialog.Green.
    """

    # show the message even if not taking a screen shot
    self.delayDisplay(description)

    if not self.enableScreenshots:
      return

    lm = slicer.app.layoutManager()
    # switch on the type to get the requested window
    widget = 0
    if type == slicer.qMRMLScreenShotDialog.FullLayout:
      # full layout
      widget = lm.viewport()
    elif type == slicer.qMRMLScreenShotDialog.ThreeD:
      # just the 3D window
      widget = lm.threeDWidget(0).threeDView()
    elif type == slicer.qMRMLScreenShotDialog.Red:
      # red slice window
      widget = lm.sliceWidget("Red")
    elif type == slicer.qMRMLScreenShotDialog.Yellow:
      # yellow slice window
      widget = lm.sliceWidget("Yellow")
    elif type == slicer.qMRMLScreenShotDialog.Green:
      # green slice window
      widget = lm.sliceWidget("Green")
    else:
      # default to using the full window
      widget = slicer.util.mainWindow()
      # reset the type so that the node is set correctly
      type = slicer.qMRMLScreenShotDialog.FullLayout

    # grab and convert to vtk image data
    qimage = ctk.ctkWidgetsUtils.grabWidget(widget)
    imageData = vtk.vtkImageData()
    slicer.qMRMLUtils().qImageToVtkImageData(qimage,imageData)

    annotationLogic = slicer.modules.annotations.logic()
    annotationLogic.CreateSnapShot(name, description, type, self.screenshotScaleFactor, imageData)

  def runTest(self):
    """
    Run a default selection of tests here.
    """
    logging.warning('No test is defined in '+self.__class__.__name__)

def createTextInput(parentCollapsibleButton, parentFormLayout, labelText, editText):
    frame = qt.QFrame(parentCollapsibleButton)
    frame.setLayout(qt.QHBoxLayout())
    parentFormLayout.addWidget(frame)
    textInputLabel = qt.QLabel(labelText, frame)
    frame.layout().addWidget(textInputLabel)
    textInput = qt.QLineEdit()
    frame.layout().addWidget(textInput)
    validator = qt.QDoubleValidator()
    validator.setNotation(0)
    textInput.setValidator(validator)
    textInput.setText(editText)
    return textInput

def init_weights(m):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_uniform_(m.weight)

class UNetDownBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetDownBlock, self).__init__()
        self.pipeline = nn.Sequential(
            # nn.Conv3d(in_size, out_size, 4, 2, 1, bias=False),
            nn.Conv3d(in_size, out_size, 4, 2, padding=1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pipeline.apply(init_weights)

    def forward(self, x):
        return self.pipeline(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_size, out_size, dropout):
        super(ResidualBlock, self).__init__()
        self.pipeline = nn.Sequential(
            nn.Conv3d(in_size, out_size, 4, 1, 1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout)
        )
        self.pipeline.apply(init_weights)

    def forward(self, x):
        x = self.pipeline(x)
        x = nn.functional.pad(x, (1, 0, 1, 0, 1, 0))
        return x

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUpBlock, self).__init__()
        self.pipeline = nn.Sequential(
            nn.ConvTranspose3d(in_size, out_size, 4, 2, padding=1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.ReLU(inplace=True),
        )
        self.pipeline.apply(init_weights)

    def forward(self, x):
        return self.pipeline(x)

class Attention_block(nn.Module):
    #Adapted from https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()

        num_features = [16, 32, 64, 64]

        self.first_layer = UNetDownBlock(in_channels, num_features[0])

        self.downs = nn.ModuleList()
        self.num_layers = len(num_features) - 1
        for i in range(self.num_layers):
            self.downs.append(UNetDownBlock(num_features[i], num_features[i + 1]))

        self.bottlenecks = nn.ModuleList()
        for i in range(4):
            self.bottlenecks.append(ResidualBlock(num_features[-1] * 2, num_features[-1], dropout=0.2))

        self.ups = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.ups.append(UNetUpBlock(num_features[-1] * 2, num_features[-1]))
            else:
                self.ups.append(UNetUpBlock(num_features[-i - 2] * 4, num_features[-i - 2]))

        self.last_layer = nn.Sequential(
            nn.ConvTranspose3d(num_features[0] * 2, out_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, oars):
        x = torch.cat((x, oars), dim=1)

        x = self.first_layer(x)
        skip_connections = []
        for d in self.downs:
            skip_connections.append(x)
            x = d(x)

        skip_connections = skip_connections[::-1]

        #         for s in skip_connections:
        #             print("skip", s.shape)

        # Middle part
        for bottleneck in self.bottlenecks:
            x_prev = x
            x = bottleneck(torch.cat((x, x_prev), dim=1))

        for i in range(len(self.ups)):
            if i == 0:
                #   print(x.shape, x_prev.shape)
                u = self.ups[i]
                concat = torch.cat((x, x_prev), dim=1)
                x = u(concat)
            else:
                # print(x.shape, skip_connections[i-1].shape)
                u = self.ups[i]
                if x.shape != skip_connections[i - 1].shape:
                    difference = np.array(skip_connections[i - 1].shape) - np.array(x.shape)
                    #        print(difference)
                    x = nn.functional.pad(x, (difference[3], 0, difference[4], 0, difference[2], 0))
                    #         print("padded", x.shape, skip_connections[i-1].shape)
                concat = torch.cat((x, skip_connections[i - 1]), dim=1)
                #      print("--", concat.shape)
                x = u(concat)

        # print(x.shape)

        x = self.last_layer(torch.cat((x, skip_connections[-1]), dim=1))

        return x

class AttentionGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        #adapated from https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
        super(AttentionGenerator, self).__init__()

        num_features = [16, 32, 64, 64]

        self.first_layer = UNetDownBlock(in_channels, num_features[0])

        self.downs = nn.ModuleList()
        self.num_layers = len(num_features) - 1
        for i in range(self.num_layers):
            self.downs.append(UNetDownBlock(num_features[i], num_features[i + 1]))

        self.bottlenecks = nn.ModuleList()
        for i in range(4):
            self.bottlenecks.append(ResidualBlock(num_features[-1] * 2, num_features[-1], dropout=0.2))

        self.ups = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.ups.append(UNetUpBlock(num_features[-1] * 2, num_features[-1]))
            else:
                self.ups.append(UNetUpBlock(num_features[-i - 2] * 4, num_features[-i - 2]))

        self.attentions = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.attentions.append(Attention_block(F_g=num_features[-1],F_l=num_features[-1], F_int=num_features[-i - 2] * 2))
            else:
                self.attentions.append(Attention_block(F_g=num_features[-i - 2] * 2, F_l=num_features[-i - 2] * 2, F_int=num_features[-i - 2]))


        self.last_layer = nn.Sequential(
            nn.ConvTranspose3d(num_features[0] * 2, out_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, cond):
        x = torch.cat((x, cond), dim=1)
        x = self.first_layer(x)
        skip_connections = []
        for d in self.downs:
            skip_connections.append(x)
            x = d(x)

        skip_connections = skip_connections[::-1]

        #         for s in skip_connections:
        #             print("skip", s.shape)

        # Middle part
        for bottleneck in self.bottlenecks:
            x_prev = x
            x = bottleneck(torch.cat((x, x_prev), dim=1))

        for i in range(len(self.ups)):
            if i == 0:
                #   print(x.shape, x_prev.shape)
                u = self.ups[i]
                attention_out = self.attentions[i](x, x_prev)
                concat = torch.cat((x, attention_out), dim=1)
                x = u(concat)
            else:
                # print(x.shape, skip_connections[i-1].shape)
                u = self.ups[i]
                if x.shape != skip_connections[i - 1].shape:
                    difference = np.array(skip_connections[i - 1].shape) - np.array(x.shape)
                    #        print(difference)
                    x = nn.functional.pad(x, (difference[3], 0, difference[4], 0, difference[2], 0))
                    #         print("padded", x.shape, skip_connections[i-1].shape)

                attention_out = self.attentions[i](x, skip_connections[i - 1])
                concat = torch.cat((x, attention_out), dim=1)
                #      print("--", concat.shape)
                x = u(concat)

        # print(x.shape)

        x = self.last_layer(torch.cat((x, skip_connections[-1]), dim=1))

        return x

def getV_X(doseVolume, oarVolume, oarCode, doseTarget):
    oar = oarVolume == oarCode
    greaterThanDose = doseVolume > doseTarget
    print("V20 debug oarCode: ", oarCode, "GreaterThanSum: ", (greaterThanDose * oar).sum(), "oarSum: ", oar.sum())

    return (greaterThanDose * oar).sum()/oar.sum() * 100

def getV_Xdiff(realVolume, estVolume, ganVolume, oarVolume, oarCode, doseTarget):
    realV_x = getV_X(realVolume, oarVolume, oarCode, doseTarget)
    estV_x = getV_X(estVolume, oarVolume, oarCode, doseTarget)
    ganV_x = getV_X(ganVolume, oarVolume, oarCode, doseTarget)

    return realV_x, estV_x, ganV_x, estV_x - realV_x, ganV_x- realV_x

def getCV(doseVolume, oarVolume, oarCode, doseTarget, sliceThickness):
    oar = oarVolume == oarCode
    lessThanDose = doseVolume < doseTarget
    volmm3 = (lessThanDose * oar).sum() * sliceThickness[0] * sliceThickness[1] * sliceThickness[2]
    # convert mm3 to cm3
    volcm3 = volmm3 / 1000
    return volcm3


def getDmaxForGivenNumVoxels(doseVolume, oarVolume, oarCode, numVoxels):
    oar = oarVolume == oarCode
    doseVolume = doseVolume * oar
    flattened_dose = doseVolume.flatten()
    sorted_array = np.sort(flattened_dose)
    return sorted_array[-numVoxels]

def getDmax(doseVolume, oarVolume, oarCode):
    oar = oarVolume == oarCode
    doseVolume = doseVolume * oar
    return doseVolume.max()