from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC, SVR
from graph_utils import shortcutToData, rankBy
import pandas as pd

def ex():
    cancer = load_breast_cancer()

    X = cancer.data[:, :2]
    y = cancer.target
    print(f'size of x is {X.size} and shape is {X.shape} and type is {type(X)}')
    print(f'size of y is {y.size} and shape is {y.shape} and type is {type(y)}')
    print(cancer.data.shape)
    # print(y)
    # print(cancer.items())
    # print(cancer.keys())
    # print(type(cancer))
    print(cancer.feature_names)
    print(cancer.target_names)



    #Build the model
    svm = SVC(kernel="rbf", gamma=0.5, C=1.0)
    # Trained the model
    svm.fit(X, y)
    print(f'prediction is {svm.predict([[5,6],[8,9],[10,15]])}')

    # Plot Decision Boundary
    DecisionBoundaryDisplay.from_estimator(
            svm,
            X[:,:2],
            response_method="predict",
            cmap=plt.cm.Spectral,
            alpha=0.8,
            xlabel=cancer.feature_names[0],
            ylabel=cancer.feature_names[1],
        )

    # Scatter plot
    plt.scatter(X[:, 0], X[:, 1], 
                c=y, 
                s=20, edgecolors="k")
    plt.show()

def test(dfs, dispNo, dispTitle):
    ranked = rankBy(dfs, (dispNo, 1), "Kernel Time", True)
    ranked["rankAsStr"] = ranked.apply(lambda y: f'{y["rank"]}', axis=1)
    ranked["shape"] = ranked.apply(lambda y: int(y["Tile Shape"] == "wide"), axis=1)
    feature_names = ["Microkernel Row Dim","UnrollAndJam Factor","UnrollAndJam Outer Loops","Microkernel Count","Regular Loads","Reused Streaming Loads","Space Needed in L1","Row Dim","Reduction Dim"]
    feature_names = ["Row Dim","Reduction Dim"]
   # feature_names=["Microkernel Count","Space Needed in L1"]
    target_name = "shape"
    train = ranked[feature_names]
  
    print(train.shape)
    X = np.array(train)
    y = np.array(ranked[target_name])
    print(f'size of x is {X.size} and shape is {X.shape} and type is {type(X)}')
    print(f'size of y is {y.size} and shape is {y.shape} and type is {type(y)}')
    # for row in train.iterrows():
    #    print(type(row))
    #    print(row[1].shape)
    #    for e in row[1]:
    #        print(e, end=' ')
    #    print('\n')



       #print(row[1])
    # print(cancer.data.shape)
    # # print(y)
    # # print(cancer.items())
    # # print(cancer.keys())
    # # print(type(cancer))
    # print(cancer.feature_names)
    # print(cancer.target_names)



    # Build the model
    svm = SVC(kernel="linear", gamma=0.5, C=1.0)
    # Trained the model
    svm.fit(X, y)

   # print(X)
    for (row,answer) in zip(X,ranked[target_name]):
        print(f'{row} {answer}')
        if svm.predict([row]) != answer:
           print(f'prediction for {row[0]}x{row[1]} is {svm.predict([row])} but it is actually {answer}')
        else:
            print(f'prediction for {row[0]}x{row[1]} is correct: {svm.predict([row])}')
    # print(f'prediction is {svm.predict([[5,6],[8,9],[10,15]])}')

    # # Plot Decision Boundary
    DecisionBoundaryDisplay.from_estimator(
            svm,
            X,
            response_method="predict",
            cmap=plt.cm.Spectral,
            alpha=0.8,
            xlabel=feature_names[0],
            ylabel=feature_names[1],
        )

    #print(X["Row Dim"].values)
    # # Scatter plot
  #  print(X)
    # plt.scatter(X[feature_names[0]], X[feature_names[1]], 
    #             c=y, 
    #             s=20, edgecolors="k")
    plt.scatter(X[:,0], X[:,1], 
                c=y, 
                s=20, edgecolors="k")
    plt.show()

def main():
    titles = [
        "Dispatch 1\nmatvec: <1x400>, <1200x400> -> <1x1200>",
        "Dispatch 8\nmatvec: <1x600>, <600x600> -> <1x600>",
        "Dispatch 7\nmatvec: <1x400>, <600x400> -> <1x600>",
    ]

    # dfs = shortcutToData("../estimated_cycles_out")
    dfs = shortcutToData("../estimated_cycles_out_2")

    test(dfs, 8, titles[1])
   
    # Load the datasets
    # cancer = load_breast_cancer()

    # X = cancer.data[:, :2]
    # y = cancer.target
    # print(f'size of x is {X.size} and shape is {X.shape} and type is {type(X)}')
    # print(f'size of y is {y.size} and shape is {y.shape} and type is {type(y)}')
    # print(cancer.data.shape)
    # # print(y)
    # # print(cancer.items())
    # # print(cancer.keys())
    # # print(type(cancer))
    # print(cancer.feature_names)
    # print(cancer.target_names)



    # #Build the model
    # svm = SVC(kernel="rbf", gamma=0.5, C=1.0)
    # # Trained the model
    # svm.fit(X, y)
    # print(f'prediction is {svm.predict([[5,6],[8,9],[10,15]])}')

    # # Plot Decision Boundary
    # DecisionBoundaryDisplay.from_estimator(
    #         svm,
    #         X[:,:2],
    #         response_method="predict",
    #         cmap=plt.cm.Spectral,
    #         alpha=0.8,
    #         xlabel=cancer.feature_names[0],
    #         ylabel=cancer.feature_names[1],
    #     )

    # # Scatter plot
    # plt.scatter(X[:, 0], X[:, 1], 
    #             c=y, 
    #             s=20, edgecolors="k")
    # plt.show()

if __name__ == "__main__":
    main()