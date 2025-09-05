import sys
import pandas as pd
import re
import os.path
# reformat old search space to conform to new conventions

if len(sys.argv) != 2:
    print("\t",end='')
    print(f"USAGE: Requires a search space csv file name")
else:
    df=pd.read_csv(sys.argv[1])
    basename = sys.argv[1][:-(len(".csv"))]
    kernelName = df.iloc[0]["Kernel Name"]
    print(kernelName)
    myRegex=re.compile(r"main\$async_dispatch_(\d+)_matmul_transpose_b_(\d+)x(\d+)x(\d+)_f64")
    d, M, N, K = myRegex.search(kernelName).groups()
    # add logistical information
    if int(M) == 1:
        df["m"]=1
    else:
        print("hello")
        df["m"]=df.apply(lambda y: y["m Dim"], axis=1)
    df["n"]=df.apply(lambda y: y["Row Dim"], axis=1)
    df["k"]=df.apply(lambda y: y["Reduction Dim"], axis=1) 
    df["M"] = int(M)  
    df["N"] = int(N)  
    df["K"] = int(K)  
    df["FakeNN JSON Name"]=df.apply(lambda y: f'{y["M"]}x{y["N"]}x{y["K"]}w{y["m"]}-{y["n"]}-{y["k"]}' ,axis=1)
    preferred_front_order = ['FakeNN JSON Name','M','N','K','m','n','k','JSON Name']
    pfoSet = set(preferred_front_order)
    wofSet = set(set(df.columns).difference(pfoSet))
    preferred_order = preferred_front_order + list(wofSet)
    df = df[preferred_order]
    df.to_csv(
            f'{basename}-logistics.csv',
            index=False,
        )
    
       