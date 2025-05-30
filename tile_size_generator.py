from dataclasses import dataclass, field
import pandas as pd
import sys
import quidditch_load_counting as qlc
from utils import InputMatrix, TileSizes, roundUpToNearestMultipleOf
# @dataclass
# class InputMatrix:
#     """Class for keeping track of matrix dimensions in"""
#     """matrix-vector transpose with type `<MxK>, <NxK> -> <MxN>` where `M = 1` (otherwise matmul)"""
#     n: int = 1200
#     k: int = 400

class TileSizeGenerator:
  def __init__(self, outputVectorEltCount, inputVectorEltCount):
    self.me = InputMatrix(n=outputVectorEltCount,k=inputVectorEltCount)

  def myfunc(self):
      print(f'I am {self.me}')
      self.validOptions(7,7)

  def validOptions(self, n_prime_min, n_prime_max):
     for i in range(8,1200+8,8):
         print(i)
         
     return 4
    # return self.me.n * self.me.k

  def weightMatTileSize(self, row_dim, reduction_dim):
    return row_dim * reduction_dim

  def spaceForTiles(self, row_dim, reduction_dim):
      # space in element count
      inputVectorTile = 1 * reduction_dim
      weightMatTiles = 2 * self.weightMatTileSize(row_dim, reduction_dim)
      space = inputVectorTile + weightMatTiles
      # space in  bytes
      spaceInBytes = space * 8 # number of elements * 8 bytes per element
      # print(f'0-{48}-{100}:\ninputVectorTile elts: {inputVectorTile} * 8 = {inputVectorTile*8}. \nweightMatTiles elts: {weightMatTiles} * 8 = {weightMatTiles*8}.\ntotal in bytes: {spaceInBytes}.')
      return spaceInBytes


  def spaceRemaining(self, row_dim, reduction_dim):
      l1MemoryBytes = 100000
      outputMatVec = roundUpToNearestMultipleOf(self.me.n, row_dim) * 8
      outputFusedAdd = self.me.n * 8
      inputFusedAdd = self.me.n * 8
      remaining = (
          l1MemoryBytes
          - outputMatVec
          - outputFusedAdd
          - inputFusedAdd
          - self.spaceForTiles(row_dim, reduction_dim)
      )
      return remaining
  # annotate a (row_dim, reduction_dim) pair with
  # total L1 space used for tiles
  # weight matrix tile size
  # total spaced used in L1
  # space remaining, etc.
  def annotateOption(self,tup):
      return (
          tup,
          self.spaceForTiles(tup[0], tup[1]),
          self.weightMatTileSize(tup[0], tup[1]),
          self.spaceRemaining(tup[0], tup[1]),
      )

  # annotate a (row_dim, reduction_dim) pair with
  # quidditch load counting information
  # flatten tuple
  # matrix-vector transpose with type `<MxK>, <NxK> -> <MxN>` where `M = 1` (otherwise matmul)
  #         
  #         outputVectorEltCount = N (AKA "row_dim")
  #         inputVectorEltCount = K (AKA "reduction dim")
  #         
  # ex. python3 tile_size_gen.py "dispatch_1" 1200 400
  def flattenThenAnnotateMore(self, ann, caseNo:int):
      input = InputMatrix(n=self.me.n,k=self.me.k)
      tiles = TileSizes(n=ann[0][0], k=ann[0][1])
      flat = self.convertAnnotationToFlatTuple(ann)
      loadInfo = (caseNo,) + qlc.getLoadCountingAnn(input,tiles) 
      concatted = flat + loadInfo
      return concatted

  # helper for converting to CSV
  def convertAnnotationToFlatTuple(self, elt):
      return (
          f"{0}-{elt[0][0]}-{elt[0][1]}",
          elt[0][0],
          elt[0][1],
          elt[1],
          elt[2],
          elt[3],
      )


  # helper for converting to CSV
  def annotationColumnNames(self):
      columns = [
          "JSON Name",
          "Row Dim",
          "Reduction Dim",
          "Space Needed in L1",
          "Weight Matrix Tile Size",
          "Space Remaining",
      ]
      return columns
  
  # export annotated options to CSV
  def exportOptionsToCSV(self, dispatchName, caseNo, options):
      flat = list(map(lambda tup: self.flattenThenAnnotateMore(tup,caseNo), options))
      cols = (self.annotationColumnNames()+["Case"])+qlc.LoadCountingAnnColumnNames()
      df = pd.DataFrame(flat, columns=cols)
      # df["Case"] = caseNo
      df.to_csv(
          f"search_spaces_out/{dispatchName}_case{caseNo}_searchSpace.csv",
          index=False,
      )
      print(
          f"Saved CSV to ./search_spaces_out/{dispatchName}_case{caseNo}_searchSpace.csv"
      )
      return df
    
def tupleIze(a,b):
  return (a,b)

def get_tile_shape(row_dim, col_dim):
    if row_dim > col_dim:
        shape = "tall"
    if row_dim < col_dim:
        shape = "wide"
    if row_dim == col_dim:
        shape = "square"
    return shape

def addMoreColsForConvenience(df):
    df["Total Loads"] = df["Regular Loads"] + df["Total Streaming Loads"]
    df["Tile Shape"] = df.apply(lambda x: get_tile_shape(x["Row Dim"], x["Reduction Dim"]), axis=1)
    return df

def main():
    args = sys.argv[1:]
    if len(args) != 6:
      print("USAGE: python3 tile_size_generator.py <N> <K> <tilesRequested.csv> <dispatchName> <CaseNo> <oracle-output-dir>")
      print(
            "\twhere N, K, refer to dimensions in a\n\tmatrix-vector transpose with type `<MxK>, <NxK> -> <MxN>` where `M = 1`"
      )
      print(
            "\tNote: when <tilesRequested.csv> is set to ORACLE_MODE, \n\t      the tile generator will automatically generate a set of valid tile sizes, \n\t      placing them in <oracle-output-dir>."
      )
      exit(1)
    if args[2] == "ORACLE_MODE":
        print("we're in oracle mode!!")
        dispatchName = args[3]
        caseNo = int(args[4])
        outputDir = args[5]
        jen = TileSizeGenerator(int(args[0]),int(args[1]))
        # sizeInfo = list(map(lambda tup: jen.annotateOption(tup), tups))
        # sizeAndLoadInfo = jen.exportOptionsToCSV(dispatchName, caseNo, sizeInfo)
        # extras = addMoreColsForConvenience(sizeAndLoadInfo)
        print(jen.myfunc())
        print(f'wrote outputs to directory {outputDir}')
        exit(0)
    jen = TileSizeGenerator(int(args[0]),int(args[1]))
    firstCSV=pd.read_csv(args[2])
    dispatchName = args[3]
    caseNo = int(args[4])
    justRuntimeInfo=firstCSV[["JSON Name","Kernel Name","Kernel Time","Total Time"]]    
    firstCSV["tuples"]=firstCSV.apply(lambda x: tupleIze(x["Row Dim"], x["Reduction Dim"]), axis=1)
    tups = firstCSV["tuples"].tolist()
    sizeInfo = list(map(lambda tup: jen.annotateOption(tup), tups))
    sizeAndLoadInfo = jen.exportOptionsToCSV(dispatchName, caseNo, sizeInfo)
    extras = addMoreColsForConvenience(sizeAndLoadInfo)
    merged = pd.merge(justRuntimeInfo,extras,on="JSON Name",how="outer")
    print(merged)
    merged.to_csv(f"search_spaces_out/{dispatchName}_case{caseNo}_everything.csv",index=False)
    #print(justTileSizes)
    # print(firstCSV["tuples"])
    # #firstCSV["size info"]=firstCSV.apply(lambda x: tester2(x["tuples"]), axis=1)
    # firstCSV["size info"]=firstCSV.apply(lambda x: tester(x["tuples"],jen), axis=1)
    # print(firstCSV["size info"])
#flat = list(map(lambda tup: flattenThenAnnotateMore(tup,caseNo), options))
    #withSizeInfo, index=justTileSizes.apply(lambda x: jen.annotateOption(x["tuples"]), axis=1)
  #       first = firstCSV[colsIWant]
  #       secondCSV=pd.read_csv(sys.argv[2])
  #       colsIWant = ["JSON Name","Kernel Name","Kernel Time","Total Time"]
  #       first = firstCSV[colsIWant]
  # print(f"merging {sys.argv[1]} {sys.argv[2]} ON {sys.argv[3]}")
  #       firstCSV=pd.read_csv(sys.argv[1])
  #       secondCSV=pd.read_csv(sys.argv[2])
  #       colsIWant = ["JSON Name","Kernel Name","Kernel Time","Total Time"]
  #       first = firstCSV[colsIWant]
  #       second = addMoreColsForConvenience(secondCSV)
  #       print(first)
  #       print(second)
  #       mergedCSV = pd.merge(first, second,on=sys.argv[3],how="outer")  
  #       print(mergedCSV)  
  #       mergedCSV.to_csv(sys.argv[4],index=False)    
    qlc.yodel()


if __name__ == "__main__":
    main()