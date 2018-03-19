import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import subprocess
#get hostname and set base directory
j = subprocess.Popen("hostname", stdout=subprocess.PIPE, shell=True)
(output, err) = j.communicate()
if 'imen' in output:
    base="/home/dnambil/Downloads/BRIDGET/plots"
else:
    base="/data/Team_Caspers/Arasan/BRIDGET/1000Brains_b1000-dMRI030_99xxxx_dwi_eddy"

print base
#read in 1000 BRAINS subject list
dfs=pd.read_table(base+'/Visit1_Age.txt',sep='\t')
#read in PSMD subjects list
dfm=pd.read_csv(base+'/TOTAL_METRICS_Skel_header.csv',sep=' ')
#merge keeps both key columns - figure out how to avoid this
dfplot=dfm.merge(dfs,left_on='NAME', right_on='Identifiers',sort=True)
df_plot=dfplot.drop('Identifiers',axis=1)
df_plot_no_visit=df_plot.drop('Visit',axis=1)
#sns.heatmap(df_plot_no_visit.corr(),xticklabels=df_plot_no_visit.corr().columns.values,yticklabels=df_plot_no_visit.corr().columns.values)

def dia_corr_mat():
    df_plot_no_visit_no_name=df_plot_no_visit.drop('NAME',axis=1)
    corr=df_plot_no_visit_no_name.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    #f, ax = plt.subplots()
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap,annot=True, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
#function to list subjects in PSMD results but not in 1000Brains list
def missing_subs():
    for i in dfm.NAME:
        if dfs[dfs.Identifiers==i]['Identifiers'].values.shape[0]<1:
            print "subject "+str(i)+" is not present in database"
            
def get_age_metric(metric):
    x=np.array([])
    y=np.array([])
    for i in dfm.NAME:
        if dfs[dfs.Identifiers==i]['Identifiers'].values.shape[0]>0:
            x=np.append(x,dfs[dfs.Identifiers==i]['Age'].values[0])		
            y=np.append(y,dfm[dfm.NAME==i][metric].values[0])
    return x,y

#plots line reg params - metric vs age
def plot_metric(metric):
    x,y=get_age_metric(metric)
    data = pd.DataFrame(data={'Age': x, metric: y})
    print x.shape
    #grid = sns.lmplot('Age', 'MD', data, size=7, truncate=True, scatter_kws={"s": 100})
    #sns.lmplot(x='Age',y='MD', data=data)
    #plt.show()
    
    #sns.regplot(x='Age',y=metric,data=data)
    sns.jointplot(x='Age',y=metric,data=data,kind='reg')

def plot_metric_2(metric):
    print metric
    sns.jointplot(x='Age',y=metric,data=df_plot_no_visit,kind='reg')
    
def scp_linreg_metric(metric):
    x,y=get_age_metric(metric)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, r_value, r_value**2, p_value

def sm_ols_metric(metric):
    x,y=get_age_metric(metric)
    model = sm.OLS(y, sm.add_constant(x)).fit()
    #predictions = model.predict(x) # make the predictions by the model
    # Print out the statistics
    print model.summary()
    
def skl_linreg_metric(metric):
    x,y=get_age_metric(metric)
    x=x.reshape(-1,1)
    y=y.reshape(-1,1)
    model=LinearRegression()
    model.fit(x,y)
    print model.score(x,y)
  
def make_df_reg_all(sd=True):    
    df_list=[]
    for i in dfm.columns.values.tolist():
        if sd:
            if "LH" in i and sd:
                slope,r,r2,p=scp_linreg_metric(i)
                row_dict={'metric':i,'slope':slope,'pearsonr':r,'r-squared':r2,'p':p}
                df_list.append(row_dict)
        else:
            if "LH" in i and "sd" not in i:
                slope,r,r2,p=scp_linreg_metric(i)
                row_dict={'metric':i,'slope':slope,'pearsonr':r,'r-squared':r2,'p':p}
                df_list.append(row_dict)
            
            #print "For "+i+":"
            #print "p-value :"+str(p)
            #print "R-squared :"+str(r2)
            #print "Slope :"+str(slope)
    df=pd.DataFrame(df_list)
    print df
    df.to_csv(base+'/linreg_metrics.csv')
   
def polyfit2(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()
    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot
    return results

#def parse_metric(metric):
#    if metric=='mean_skel_MD_LH_RH':
#        return mean_MD
#    elif metric=='sd_skel_MD_LH_RH':
#        return sd_MD    .
    
#def df_corr_all():
#    df_corr_list=[]
#    x=np.array([])
#    y=np.array([])
#    for i in dfm.NAME:
#        if dfs[dfs.Identifiers==i]['Identifiers'].values.shape[0]>0:
#            x=np.append(x,dfs[dfs.Identifiers==i]['Age'].values[0])		
#            for j in dfm.columns.values.tolist():
#                if "LH" in j:
#                    y=np.append(y,dfm[dfm.NAME==i][j].values[0])
#
#                    
#            y=np.append(y,dfm[dfm.NAME==i][metric].values[0])
#    
#    Age,_ = get_age_metric()
#    for i in dfm.NAME:
#        if 
#        row_dict={'Age':}