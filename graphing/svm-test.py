from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC, SVR
from graph_utils import shortcutToData, rankBy, Graph2D, Keys2D, CustomMarker, graphEmAll
import pandas as pd
from PIL import Image

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

def shapeTest(dfs, dispNo, dispTitle):
    ranked = rankBy(dfs, (dispNo, 1), "Kernel Time", True)
    ranked["rankAsStr"] = ranked.apply(lambda y: f'{y["rank"]}', axis=1)
    ranked["shape"] = ranked.apply(lambda y: int(y["Tile Shape"] == "wide"), axis=1)
    feature_names = ["Microkernel Row Dim","UnrollAndJam Factor","UnrollAndJam Outer Loops","Microkernel Count","Regular Loads","Reused Streaming Loads","Space Needed in L1","Row Dim","Reduction Dim"]
    feature_names = ["Row Dim","Reduction Dim"]
    target_name = "shape"
    train = ranked[feature_names]
  
    #print(train.shape)
    X = np.array(train)
    y = np.array(ranked[target_name])
    # print(f'size of x is {X.size} and shape is {X.shape} and type is {type(X)}')
    # print(f'size of y is {y.size} and shape is {y.shape} and type is {type(y)}')
   
    # Build the model
    svm = SVC(kernel="linear", gamma=0.5, C=1.0)
    # Train the model
    svm.fit(X, y)

    # for (row,answer) in zip(X,ranked[target_name]):
    #     print(f'{row} {answer}')
    #     if svm.predict([row]) != answer:
    #        print(f'prediction for {row[0]}x{row[1]} is {svm.predict([row])} but it is actually {answer}')
    #     else:
    #         print(f'prediction for {row[0]}x{row[1]} is correct: {svm.predict([row])}')

    # Plot Decision Boundary
    DecisionBoundaryDisplay.from_estimator(
            svm,
            X,
            response_method="predict",
            cmap=plt.cm.Spectral,
            alpha=0.8,
            xlabel=feature_names[0],
            ylabel=feature_names[1],
        )
    # plot the points
    plt.scatter(X[:,0], X[:,1], 
                c=y, 
                s=20, edgecolors="k")
    plt.show()

def createTableGraph(data,dispNo,dispTitle):
    tableData = data[["rankAsStr","Microkernel Row Dim","UnrollAndJam Factor","UnrollAndJam Outer Loops","Microkernel Count","Regular Loads","Reused Streaming Loads","Space Needed in L1","Row Dim","Reduction Dim"]]
    colLabels = ["rank","n'","U&J","CC OLs","Micro Runs","RLs","Reused SLs","L1 Usage","n","k"]
    defW = (1/(len(colLabels)*3)) # default width
    #print(defW)
    tableColWidths = [defW*0.6,defW*0.5,defW*0.4,defW,defW*1.25,defW,defW*1.5,defW*1.6,defW*0.5,defW*0.5]#[defW]*len(colLabels)
    b = Graph2D(
        imagePath=f'dispatch-{dispNo}-case-{1}',
        keys=Keys2D(
            x="rank",
            x_label="Rank",
            x_unit="fastest to slowest",
            y="Kernel Time",
            y_label="Kernel Time",
            y_unit="cycles",
        ),
        title=dispTitle,
        scatterSets=[
            (
                data,#dfs[(dispNo, 1)],
                CustomMarker(
                    y="Kernel Time",
                    label= lambda y: f'    {y["JSON Name"]}', 
                    # label= lambda y: f'{y["JSON Name"]} ({y["Microkernel Row Dim"]}, UaJ = {y["UnrollAndJam Factor"]}, {y["UnrollAndJam Outer Loops"]} outer loops)',                   
                    marker=lambda x: f'${x["rank"]}$',#f'$({x["UnrollAndJam Factor"]},{int(x["UnrollAndJam Outer Loops"])})$',
                    size=lambda y=0: (plt.rcParams["lines.markersize"] ** 2)*2,
                    stroke=lambda x: 'Black',#"Purple",
                    fill=lambda x: 'Black'#"Purple",
                ),
            ),
            (
                data,#dfs[(dispNo, 1)],
                CustomMarker(
                    y="Predicted Kernel Time",
                    label= lambda y: f'    {y["JSON Name"]}', 
                    # label= lambda y: f'{y["JSON Name"]} ({y["Microkernel Row Dim"]}, UaJ = {y["UnrollAndJam Factor"]}, {y["UnrollAndJam Outer Loops"]} outer loops)',                   
                    marker=lambda x: f'${x["rank"]}$',#f'$({x["UnrollAndJam Factor"]},{int(x["UnrollAndJam Outer Loops"])})$',
                    size=lambda y=0: (plt.rcParams["lines.markersize"] ** 2)*2,
                    stroke=lambda x: "Purple",
                    fill=lambda x: "Purple",
                ),
            )
        ],
        legend = False,
        table = True,
        table_pos="right",
        table_bb=(1.01,0,1,1), #self.scale(rw / w, rh / h)
        table_col_widths = tableColWidths,
        table_col_labels=colLabels,
        table_row_labels=[],
        table_data=tableData
    )
    return b

def learnCostTest(dfs, dispNo, dispTitle):
    ranked = rankBy(dfs, (dispNo, 1), "Kernel Time", True)
    ranked["rankAsStr"] = ranked.apply(lambda y: f'{y["rank"]}', axis=1)
    feature_names = ["Microkernel Count","Regular Loads","Reused Streaming Loads","Space Needed in L1","Row Dim","Reduction Dim"]
    target_name = "Kernel Time"
    X = np.array(ranked[feature_names])
    y = np.array(ranked[target_name])
    print(f'size of x is {X.size} and shape is {X.shape} and type is {type(X)}')
    print(f'size of y is {y.size} and shape is {y.shape} and type is {type(y)}')
   
    # Build the model
    svm = SVR(kernel="linear", gamma=0.5, C=1.0) # maybe try poly or rbf?
    # Train the model
    svm.fit(X, y)
    print(svm._decision_function)

    ranked["Predicted Kernel Time"] = ranked.apply(lambda y: svm.predict([y[["Microkernel Count","Regular Loads","Reused Streaming Loads","Space Needed in L1","Row Dim","Reduction Dim"]]])[0], axis=1)
    ranked = ranked.sort_values("Predicted Kernel Time", ascending=True)
    #print(ranked[["Kernel Time", "Predicted Kernel Time","rank"]])

    # for (row,answer) in zip(X,ranked[target_name]):
    #     #print(f'{row} {answer}')
    #     if svm.predict([row]) != answer:
    #        print(f'prediction for {row[0]}x{row[1]} is {svm.predict([row])} but it is actually {answer}')
    #     else:
    #         print(f'prediction for {row[0]}x{row[1]} is correct: {svm.predict([row])}')

    # [1/(len(g.table_col_labels))*0.25]*len(g.table_col_labels)
    #[1/(len(g.table_col_labels))*0.25]*len(g.table_col_labels)
    b = createTableGraph(ranked,dispNo,dispTitle)
    # now let's use model trained on dispatch 8 to predict dispatch 7 and dispatch 1!!
    dispatch_7 = rankBy(dfs, (7, 1), "Kernel Time", True)
    dispatch_7["rankAsStr"] = dispatch_7.apply(lambda y: f'{y["rank"]}', axis=1)
    dispatch_7["Predicted Kernel Time"] = dispatch_7.apply(lambda y: svm.predict([y[["Microkernel Count","Regular Loads","Reused Streaming Loads","Space Needed in L1","Row Dim","Reduction Dim"]]])[0], axis=1)
    c = createTableGraph(dispatch_7,7,"Dispatch 7\nmatvec: <1x400>, <600x400> -> <1x600>")
    dispatch_1 = rankBy(dfs, (1, 1), "Kernel Time", True)
    dispatch_1["rankAsStr"] = dispatch_1.apply(lambda y: f'{y["rank"]}', axis=1)
    dispatch_1["Predicted Kernel Time"] = dispatch_1.apply(lambda y: svm.predict([y[["Microkernel Count","Regular Loads","Reused Streaming Loads","Space Needed in L1","Row Dim","Reduction Dim"]]])[0], axis=1)
    #print("dispatch 1 ")
    #print(dispatch_1[["Kernel Time", "Predicted Kernel Time","rank"]])
    d = createTableGraph(dispatch_1,1,"Dispatch 1\nmatvec: <1x400>, <1200x400> -> <1x1200>")
    gs = []
    gs.append(b)
    gs.append(c)
    gs.append(d)
    return gs


def learnOverheadTest(dfs, dispNo, dispTitle):
    ranked = rankBy(dfs, (dispNo, 1), "Kernel Time", True)
    ranked["rankAsStr"] = ranked.apply(lambda y: f'{y["rank"]}', axis=1)
    ranked["Hardware Loop Time Estimate"] = ranked.apply(lambda y: y["Kernel Time Estimate"]/y["Microkernel Count"], axis=1)
    #feature_names = ["Hardware Loop Time Estimate","UnrollAndJam Outer Loops","Microkernel Count"]
    feature_names = ["Kernel Time Estimate","UnrollAndJam Outer Loops"]
    target_name = "Kernel Time"
    train = ranked[feature_names]
  
    #print(train.shape)
    X = np.array(train)
    y = np.array(ranked[target_name])
    # print(f'size of x is {X.size} and shape is {X.shape} and type is {type(X)}')
    # print(f'size of y is {y.size} and shape is {y.shape} and type is {type(y)}')
   
    # Build the model
    svm = SVR(kernel="linear", gamma=0.5, C=1.0)
    # Train the model
    svm.fit(X, y)

    print(svm._decision_function)
    print(svm.coef_)

    for (row,answer) in zip(X,ranked[target_name]):
       # print(f'{row} {answer}')
        if svm.predict([row]) != answer:
           print(f'prediction for {row[0]}x{row[1]} is {svm.predict([row])} but it is actually {answer}')

    ranked["Predicted Kernel Time"] = ranked.apply(lambda y: svm.predict([y[feature_names]])[0], axis=1)
    ranked = ranked.sort_values("Predicted Kernel Time", ascending=True)
    b = createTableGraph(ranked,dispNo,dispTitle)
    
    # now let's use model trained on dispatch 8 to predict dispatch 7 and dispatch 1!!
    dispatch_7 = rankBy(dfs, (7, 1), "Kernel Time", True)
    dispatch_7["rankAsStr"] = dispatch_7.apply(lambda y: f'{y["rank"]}', axis=1)
    dispatch_7["Hardware Loop Time Estimate"] = ranked.apply(lambda y: y["Kernel Time Estimate"]/y["Microkernel Count"], axis=1)
    dispatch_7["Predicted Kernel Time"] = dispatch_7.apply(lambda y: svm.predict([y[feature_names]])[0], axis=1)
    c = createTableGraph(dispatch_7,7,"Dispatch 7\nmatvec: <1x400>, <600x400> -> <1x600>")
   
    dispatch_1 = rankBy(dfs, (1, 1), "Kernel Time", True)
    dispatch_1["rankAsStr"] = dispatch_1.apply(lambda y: f'{y["rank"]}', axis=1)
    dispatch_1["Hardware Loop Time Estimate"] = ranked.apply(lambda y: y["Kernel Time Estimate"]/y["Microkernel Count"], axis=1)
    dispatch_1["Predicted Kernel Time"] = dispatch_1.apply(lambda y: svm.predict([y[feature_names]])[0], axis=1)
    d = createTableGraph(dispatch_1,1,"Dispatch 1\nmatvec: <1x400>, <1200x400> -> <1x1200>")
    
    gs = []
    gs.append(b)
    gs.append(c)
    gs.append(d)
    return gs

def svm_find_overhead_constant(titles,dfs):
    gs = learnOverheadTest(dfs, 8, titles[1])
    for g in gs:
            graphEmAll((1, 1), [g])
            top = Image.open('context2.png') # hard coded
            bot = Image.open(f'{g.imagePath}.png')    
            resized = top.copy()
            resized.thumbnail(bot.size, Image.Resampling.LANCZOS)
            resized.save("resized.png")
            top = Image.open('resized.png')
            canvas = Image.new('RGBA', (bot.size[0], bot.size[1]+top.size[1]), (0, 0, 0, 0))
            canvas.paste(resized, (0, 0), resized)
            canvas.paste(bot, (0, top.size[1]), bot)
            canvas.save('Image.png')
    # put all the graphs in one image
    top = Image.open('resized.png')
    g0 = Image.open(f'{gs[0].imagePath}.png')  
    g1 = Image.open(f'{gs[1].imagePath}.png')  
    g2 = Image.open(f'{gs[2].imagePath}.png')  
    canvas = Image.new('RGBA', (g2.size[0], top.size[1]+g0.size[1]+g1.size[1]+g2.size[1]), (0, 0, 0, 0))
    canvas.paste(top, (0, 0), top)
    canvas.paste(g0, (0, top.size[1]), g0)
    canvas.paste(g1, (0, top.size[1]+g0.size[1]), g1)
    canvas.paste(g2, (0, top.size[1]+g0.size[1]+g1.size[1]), g2)
    canvas.save('3-dispatches-svm-overhead-constant.png')
    
def svm_disp_8_prediction_graphs(titles,dfs):
    gs = learnCostTest(dfs, 8, titles[0])
    for g in gs:
            graphEmAll((1, 1), [g])
            top = Image.open('context2.png') # hard coded
            bot = Image.open(f'{g.imagePath}.png')    
            resized = top.copy()
            resized.thumbnail(bot.size, Image.Resampling.LANCZOS)
            resized.save("resized.png")
            top = Image.open('resized.png')
            canvas = Image.new('RGBA', (bot.size[0], bot.size[1]+top.size[1]), (0, 0, 0, 0))
            canvas.paste(resized, (0, 0), resized)
            canvas.paste(bot, (0, top.size[1]), bot)
            canvas.save('Image.png')
    # put all the graphs in one image
    top = Image.open('resized.png')
    g0 = Image.open(f'{gs[0].imagePath}.png')  
    g1 = Image.open(f'{gs[1].imagePath}.png')  
    g2 = Image.open(f'{gs[2].imagePath}.png')  
    canvas = Image.new('RGBA', (g2.size[0], top.size[1]+g0.size[1]+g1.size[1]+g2.size[1]), (0, 0, 0, 0))
    canvas.paste(top, (0, 0), top)
    canvas.paste(g0, (0, top.size[1]), g0)
    canvas.paste(g1, (0, top.size[1]+g0.size[1]), g1)
    canvas.paste(g2, (0, top.size[1]+g0.size[1]+g1.size[1]), g2)
    canvas.save('3-dispatches-svm.png')


def main():
    titles = [
        "Dispatch 8\nmatvec: <1x600>, <600x600> -> <1x600>",
        "Dispatch 7\nmatvec: <1x400>, <600x400> -> <1x600>",
        "Dispatch 1\nmatvec: <1x400>, <1200x400> -> <1x1200>",
    ]

    dfs = shortcutToData("../estimated_cycles_no_overhead")

    #svm_disp_8_prediction_graphs(titles,dfs)
    svm_find_overhead_constant(titles,dfs)
    

if __name__ == "__main__":
    main()